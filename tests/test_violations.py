import json

import numpy as np
import pytest

from traffic_agent.adapters.alerts.telegram import TelegramAlerter
from traffic_agent.config import AppConfig, TelegramCfg
from traffic_agent.core.events import Event, EventBus, EventType
from traffic_agent.core.evidence import EvidenceStore
from traffic_agent.core.pipeline import Pipeline
from traffic_agent.core.types import ApproachState, Track, WorldState
from traffic_agent.enforcement.challan import make_challan_pdf
from traffic_agent.enforcement.violations import ViolationModule
from traffic_agent.incidents.collision import CollisionModule, touching
from traffic_agent.perception.light import classify_light
from traffic_agent.perception.plates import ALPHABET, decode_ocr
from traffic_agent.perception.zones import Zones
from traffic_agent.storage.db import Database

ZONES = {"px_per_m": 10, "metric": True,
         "lanes": [{"name": "inbound", "inbound": True, "polygon": [[0, 0], [200, 0], [200, 50], [0, 50]],
                    "direction": [[0, 25], [200, 25]]}],
         "stop_line": [[150, 0], [150, 50]]}


def _collect(bus, *types):
    out = []
    bus.subscribe(lambda e: out.append(e) if e.type in types else None)
    return out


def _track(tid=1, **kw):
    t = Track(track_id=tid, box=np.array([10, 10, 50, 40], np.float32), cls=kw.pop("cls", "car"),
              score=0.9, confirmed=True, in_roi=True)
    for k, v in kw.items():
        if k == "attrs":
            t.attrs.update(v)
        else:
            setattr(t, k, v)
    return t


def _world(t, tracks, signal="RED", a="N"):
    return WorldState(t=t, approaches={a: ApproachState(a, t, tracks=tracks)}, signal={a: signal})


def _module(tmp_path, bus):
    cfg = AppConfig()
    ev = EvidenceStore(tmp_path / "ev")
    return ViolationModule(cfg, bus, {"N": Zones.from_dict(ZONES), "S": Zones.from_dict(ZONES)}, ev), ev


# ---------------------------------------------------------------- violation rules
def test_red_light_needs_grace_period_and_exempts_ambulance(tmp_path):
    bus = EventBus()
    got = _collect(bus, EventType.VIOLATION)
    m, _ = _module(tmp_path, bus)
    m.update(_world(10.0, [], "RED"), {})                                      # red starts at t=10
    m.update(_world(10.5, [_track(1, crossed_stop_line=True)], "RED"), {})     # 0.5 s in: amber-runner, ok
    m.update(_world(13.0, [_track(2, crossed_stop_line=True, attrs={"emergency": True})], "RED"), {})
    assert got == []
    m.update(_world(13.0, [_track(3, crossed_stop_line=True)], "RED"), {})
    assert [e.data["violation"] for e in got] == ["red_light"]


def test_same_plate_seen_by_two_cameras_is_fined_once(tmp_path):
    bus = EventBus()
    got = _collect(bus, EventType.VIOLATION)
    m, ev = _module(tmp_path, bus)
    bike = dict(cls="motorcycle", attrs={"helmet": False, "plate": "TS09EC6993"})
    for k in range(5):
        w = WorldState(t=1.0 + k * 0.1, signal={"N": "GREEN", "S": "GREEN"}, approaches={
            "N": ApproachState("N", 0, tracks=[_track(4, **bike)]),
            "S": ApproachState("S", 0, tracks=[_track(9, **bike)])})
        m.update(w, {})
    ev.flush()
    assert len(got) == 1 and got[0].data["plate"] == "TS09EC6993"


def test_wrong_way_and_speeding(tmp_path):
    bus = EventBus()
    got = _collect(bus, EventType.VIOLATION)
    m, _ = _module(tmp_path, bus)
    hist = [(0, 0, 0)] * 3
    for k in range(15):
        t = k * 0.1
        m.update(_world(t, [_track(5, heading_ok=False, speed_kmph=20, history=hist),
                            _track(6, heading_ok=True, speed_kmph=80, history=hist),
                            _track(7, heading_ok=True, speed_kmph=50, history=hist)], "GREEN"), {})
    assert sorted((e.data["violation"], e.data["track_id"]) for e in got) == [("speeding", 6), ("wrong_way", 5)]


def test_speeding_needs_real_metres(tmp_path):
    bus = EventBus()
    got = _collect(bus, EventType.VIOLATION)
    cfg = AppConfig()
    uncalibrated = Zones.from_dict({k: v for k, v in ZONES.items() if k != "metric"})
    m = ViolationModule(cfg, bus, {"N": uncalibrated}, EvidenceStore(tmp_path))
    for k in range(15):
        m.update(_world(k * 0.1, [_track(6, speed_kmph=90, history=[(0, 0, 0)] * 3)], "GREEN"), {})
    assert got == [] and m.capabilities()["N"]["speeding"] is False


def test_video_mode_red_light_only_with_real_light_in_view(tmp_path):
    cfg = AppConfig(mode="video", sources={"N": {"path": "x.mp4"}})
    m = ViolationModule(cfg, EventBus(), {"N": Zones.from_dict(ZONES)}, EvidenceStore(tmp_path))
    assert m.capabilities()["N"]["red_light"] is False
    m2 = ViolationModule(cfg, EventBus(), {"N": Zones.from_dict(dict(ZONES, light_roi=[0, 0, 10, 30]))},
                         EvidenceStore(tmp_path))
    assert m2.capabilities()["N"]["red_light"] is True


def test_full_frame_fallback_zones_never_fine_wrong_way(tmp_path):
    bus = EventBus()
    got = _collect(bus, EventType.VIOLATION)
    m = ViolationModule(AppConfig(), bus, {"N": Zones.full_frame(640, 360)}, EvidenceStore(tmp_path))
    for k in range(20):
        m.update(_world(k * 0.1, [_track(5, heading_ok=False, speed_kmph=30)], "GREEN"), {})
    assert got == [] and m.capabilities()["N"]["wrong_way"] is False


def test_challan_pdf_and_fine_row(tmp_path):
    bus = EventBus()
    db = Database(tmp_path / "t.db")
    bus.subscribe(db.on_event)
    fines = _collect(bus, EventType.FINE_ISSUED)
    m, ev = _module(tmp_path, bus)
    frame = np.full((60, 220, 3), 80, np.uint8)
    m.update(_world(0.0, [], "RED"), {})
    m.update(_world(5.0, [_track(3, crossed_stop_line=True, attrs={"plate": "TS01AB1234"})], "RED"), {"N": frame})
    ev.flush()
    assert len(fines) == 1
    pdf = ev.abs(fines[0].evidence[0])
    assert pdf.read_bytes()[:4] == b"%PDF"
    rows = db.recent_fines()
    assert rows[0]["plate"] == "TS01AB1234" and rows[0]["amount"] == 1000 and rows[0]["status"] == "ISSUED"
    assert db.set_fine_status(rows[0]["challan_no"], "PAID")
    assert db.recent_fines(plate="ts01ab1234")[0]["status"] == "PAID"
    assert db.fine_summary()["by_violation"] == {"red_light": 1}


def test_challan_without_plate_goes_to_review(tmp_path):
    p = make_challan_pdf(tmp_path / "c.pdf", {"challan_no": "X1", "ts": 0, "violation": "speeding",
                                              "amount": 2000, "plate": None, "plate_status": "REVIEW"})
    assert p.read_bytes()[:4] == b"%PDF"


# ---------------------------------------------------------------- collision
def _ctrack(tid, x, speed):
    return Track(track_id=tid, box=np.array([x, 10, x + 40, 30], np.float32), cls="car", score=0.9,
                 confirmed=True, speed_kmph=speed)


class _FakeAlerter:
    def __init__(self):
        self.sent = []

    def send(self, text, photo=None, lat=None, lon=None, stem=""):
        self.sent.append(text)
        return {"delivered": True, "channel": "fake"}

    def describe(self):
        return "fake"


def _run_collision(speeds_a, speeds_b, xb, tmp_path, approaches=("N",)):
    bus = EventBus()
    al = _FakeAlerter()
    ev = EvidenceStore(tmp_path)
    m = CollisionModule(AppConfig(), bus, ev, al)
    got = _collect(bus, EventType.COLLISION, EventType.AMBULANCE_CALLED)
    for k, (va, vb) in enumerate(zip(speeds_a, speeds_b)):
        aps = {a: ApproachState(a, 0, tracks=[_ctrack(1, 0, va), _ctrack(2, xb, vb)]) for a in approaches}
        m.update(WorldState(t=k * 0.1, approaches=aps, signal={}), {})
    ev.flush()
    return got, al


def test_sudden_stop_with_contact_is_a_collision(tmp_path):
    crash = [40.0] * 10 + [0.0] * 40
    got, al = _run_collision(crash, [0.0] * 50, 41, tmp_path)
    assert [e.type for e in got] == [EventType.COLLISION, EventType.AMBULANCE_CALLED]
    assert len(al.sent) == 1 and "ACCIDENT" in al.sent[0]


def test_gradual_stop_in_a_queue_is_not_a_collision(tmp_path):
    braking = list(np.linspace(40, 0, 40)) + [0.0] * 40    # 4 s to stop
    assert _run_collision(braking, [0.0] * 80, 41, tmp_path)[0] == []
    crash = [40.0] * 10 + [0.0] * 40
    assert _run_collision(crash, [0.0] * 50, 70, tmp_path)[0] == []   # stopped, but a gap between them


def test_one_crash_on_two_cameras_calls_one_ambulance(tmp_path):
    crash = [40.0] * 10 + [0.0] * 40
    got, al = _run_collision(crash, [0.0] * 50, 41, tmp_path, approaches=("N", "E"))
    assert len(al.sent) == 1
    assert sum(1 for e in got if e.type == EventType.COLLISION and e.data.get("merged")) == 1


def test_touching():
    a = np.array([0, 0, 40, 20], np.float32)
    assert touching(a, np.array([41, 0, 80, 20], np.float32), 0.05)
    assert not touching(a, np.array([50, 0, 90, 20], np.float32), 0.05)


def test_telegram_dry_run_writes_record(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    ev = EvidenceStore(tmp_path)
    res = TelegramAlerter(TelegramCfg(), ev).send("hello", lat=1.0, lon=2.0, stem="alert_x")
    assert res["channel"] == "dry-run" and not res["delivered"]
    assert json.loads(ev.abs(res["record"]).read_text())["text"] == "hello"


# ---------------------------------------------------------------- perception helpers
def test_plate_ocr_decoding_drops_padding():
    probs = np.zeros((10, len(ALPHABET)), np.float32)
    for k, ch in enumerate("TS09EC6993"[:8] + "__"):
        probs[k, ALPHABET.index(ch)] = 0.9
    assert decode_ocr(probs) == ("TS09EC69", pytest.approx(0.9))


def test_light_colour_classifier():
    def lamp(bgr):
        img = np.zeros((30, 10, 3), np.uint8)
        img[10:20] = bgr
        return img
    assert classify_light(lamp((40, 40, 240))) == "RED"
    assert classify_light(lamp((60, 220, 60))) == "GREEN"
    assert classify_light(lamp((30, 30, 30))) is None


# ---------------------------------------------------------------- end to end in the simulator
def test_every_demo_event_is_caught_and_normal_traffic_is_clean():
    cfg = AppConfig()
    cfg.sim.realtime = False
    cfg.sim.emergency_per_hour = 0
    bus = EventBus()
    got = _collect(bus, EventType.VIOLATION, EventType.COLLISION, EventType.FINE_ISSUED)
    p = Pipeline(cfg, bus)
    for _ in range(1200):                        # 2 minutes of normal traffic
        p.tick()
    false = [e for e in got if e.type == EventType.COLLISION or e.data.get("violation") not in (None, "no_helmet")]
    assert false == [], [e.message for e in false]
    got.clear()
    for kind in ("red_light", "wrong_way", "no_helmet", "speeding", "crash"):
        p.inject_demo(kind)
        for _ in range(500 if kind == "speeding" else 250):   # the speeder may wait for a free green
            p.tick()
    p.evidence.flush()
    caught = {e.data.get("violation") for e in got if e.type == EventType.VIOLATION}
    assert {"red_light", "wrong_way", "no_helmet", "speeding"} <= caught, [e.message for e in got]
    assert any(e.type == EventType.COLLISION for e in got)
    assert any(e.type == EventType.FINE_ISSUED for e in got)
