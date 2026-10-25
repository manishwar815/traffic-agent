"""Traffic violations + e-challan.

Rules (each per confirmed track, each fined at most once per track):
  red_light  crossed the stop line while its light had been RED for > red_grace_s.
             Sim: our own signal. Video: our signal is virtual (recorded drivers
             never saw it), so only approaches with a real light in view
             (zones "light_roi") are checked, against that light.
  wrong_way  moving against its lane's legal direction for wrong_way_min_s.
             Needs drawn lanes: off with the full-frame fallback zones.
  no_helmet  motorcycle whose helmet classifier said "no helmet" in
             helmet_min_frames frames. Sim: the simulator's stand-in classifier.
             Video: needs a helmet model (attrs["helmet"]); off otherwise.
  speeding   above speed_limit * (1 + tolerance) for speed_min_s. Needs real
             metres: sim cameras, or a calibrated zones file ("calib").

Emergency vehicles on duty (attrs["emergency"]) are exempt from red light and speeding.

For each violation: marked snapshot -> VIOLATION event (immediately), then on
the evidence worker: close-up, plate read (video), challan PDF -> FINE_ISSUED
event (the database turns that into a row in `fines`).
"""
from __future__ import annotations

import itertools
import time
from typing import Optional

from ..config import AppConfig, resolve
from ..core.events import Event, EventBus, EventType, Severity
from ..core.evidence import EvidenceStore
from ..core.types import Track, WorldState
from ..perception.light import LightReader
from .challan import VIOLATION_TITLES, make_challan_pdf

TAGS = {"red_light": "RED-LIGHT", "wrong_way": "WRONG-WAY", "no_helmet": "NO-HELMET", "speeding": "SPEEDING"}
_seq = itertools.count(1)


class ViolationModule:
    def __init__(self, cfg: AppConfig, bus: EventBus, zones: dict, evidence: EvidenceStore,
                 plate_reader=None) -> None:
        self.app = cfg
        self.cfg = cfg.violations
        self.bus = bus
        self.zones = zones                          # approach -> Zones (filled lazily by the pipeline)
        self.evidence = evidence
        self.plates = plate_reader                  # PlateReader or None (sim supplies plates)
        self.sim = cfg.mode == "sim"
        self.red_since: dict[str, float] = {}
        self._last_sig: dict[str, str] = {}
        self._lights: dict[str, LightReader] = {}
        self._issued: set[tuple[str, int, str]] = set()
        self._recent_plate: dict[tuple[str, str], float] = {}
        self._wrong_since: dict[tuple[str, int], float] = {}
        self._over_since: dict[tuple[str, int], float] = {}
        self._max_speed: dict[tuple[str, int], float] = {}
        self._no_helmet: dict[tuple[str, int], int] = {}
        self._seen: dict[tuple[str, int], float] = {}
        self.counts = {k: 0 for k in TAGS}
        self.fines_total = 0
        self.recent: list[dict] = []

    # ------------------------------------------------------------------ what can be checked
    def _rule_on(self, name: str) -> bool:
        return self.cfg.enabled and getattr(self.cfg, name).enabled

    def _metric(self, a: str) -> bool:
        z = self.zones.get(a)
        return z is not None and (z.calib_h is not None or bool(z.raw.get("metric")))

    def _lanes_drawn(self, a: str) -> bool:
        z = self.zones.get(a)
        return z is not None and bool(z.lanes) and not z.raw.get("fallback")

    def _light(self, a: str, world: WorldState, frames: dict) -> Optional[str]:
        """The light the drivers on approach a actually see."""
        if self.sim:
            return world.signal.get(a)
        z = self.zones.get(a)
        if z is None or not z.light_roi:
            return None
        if a not in self._lights:
            self._lights[a] = LightReader(z.light_roi)
        if a in frames:
            self._lights[a].update(frames[a])
        return self._lights[a].state

    def capabilities(self) -> dict:
        """Which rules are live per approach (shown on the dashboard)."""
        out = {}
        for a in self.zones:
            out[a] = {
                "red_light": self._rule_on("red_light") and (self.sim or bool(self.zones[a].light_roi)),
                "wrong_way": self._rule_on("wrong_way") and self._lanes_drawn(a),
                "no_helmet": self._rule_on("no_helmet"),   # needs a helmet classifier in video mode
                "speeding": self._rule_on("speeding") and self._metric(a),
            }
        return out

    # ------------------------------------------------------------------ main
    def update(self, world: WorldState, frames: dict) -> None:
        if not self.cfg.enabled:
            return
        t = world.t
        c = self.cfg
        limit = c.speed_limit_kmph * (1 + c.speed_tolerance_pct / 100.0)
        for a, st in world.approaches.items():
            light = self._light(a, world, frames)
            if light != self._last_sig.get(a):
                self._last_sig[a] = light
                if light == "RED":
                    self.red_since[a] = t
            metric = self._metric(a)
            lanes_drawn = self._lanes_drawn(a)
            for tr in st.tracks:
                if not tr.confirmed:
                    continue
                key = (a, tr.track_id)
                self._seen[key] = t
                on_duty = bool(tr.attrs.get("emergency"))

                if (self._rule_on("red_light") and tr.crossed_stop_line and light == "RED" and not on_duty
                        and t - self.red_since.get(a, t) >= c.red_grace_s):
                    self._violation("red_light", a, tr, world, frames,
                                    {"Signal": f"RED for {t - self.red_since[a]:.1f} s when crossing",
                                     "Speed at stop line": f"{tr.speed_kmph:.0f} km/h"})

                if self._rule_on("wrong_way") and lanes_drawn:
                    if tr.heading_ok is False and tr.speed_kmph >= c.min_speed_kmph:
                        since = self._wrong_since.setdefault(key, t)
                        if t - since >= c.wrong_way_min_s:
                            self._violation("wrong_way", a, tr, world, frames,
                                            {"Lane": tr.attrs.get("lane") or "-",
                                             "Against traffic for": f"{t - since:.1f} s"})
                    elif tr.heading_ok:
                        self._wrong_since.pop(key, None)

                if (self._rule_on("no_helmet") and tr.cls == "motorcycle" and tr.attrs.get("helmet") is False
                        and (tr.in_roi or tr.attrs.get("was_inbound"))):
                    self._no_helmet[key] = self._no_helmet.get(key, 0) + 1
                    if self._no_helmet[key] >= c.helmet_min_frames:
                        self._violation("no_helmet", a, tr, world, frames,
                                        {"Helmet": f"not detected in {self._no_helmet[key]} frames"})

                if self._rule_on("speeding") and metric and not on_duty:
                    if tr.speed_kmph > limit and len(tr.history) >= 3:
                        since = self._over_since.setdefault(key, t)
                        self._max_speed[key] = max(self._max_speed.get(key, 0.0), tr.speed_kmph)
                        if t - since >= c.speed_min_s:
                            self._violation("speeding", a, tr, world, frames,
                                            {"Measured speed": f"{self._max_speed[key]:.0f} km/h",
                                             "Speed limit": f"{c.speed_limit_kmph:.0f} km/h"})
                    else:
                        self._over_since.pop(key, None)
        self._prune(t)

    def _prune(self, t: float) -> None:
        if len(self._seen) < 400:
            return
        old = [k for k, ts in self._seen.items() if t - ts > 30]
        for k in old:
            for d in (self._seen, self._wrong_since, self._over_since, self._max_speed, self._no_helmet):
                d.pop(k, None)
        self._recent_plate = {k: v for k, v in self._recent_plate.items() if t - v < self.cfg.dedup_s}

    # ------------------------------------------------------------------ one violation
    def _violation(self, kind: str, a: str, tr: Track, world: WorldState, frames: dict, details: dict) -> None:
        ikey = (a, tr.track_id, kind)
        if ikey in self._issued:
            return
        self._issued.add(ikey)
        t = world.t
        plate = tr.attrs.get("plate")                           # sim: stand-in ANPR
        if plate:
            pkey = (plate, kind)
            if t - self._recent_plate.get(pkey, -1e9) < self.cfg.dedup_s:
                return                                          # same vehicle seen by another camera
            self._recent_plate[pkey] = t
        tr.attrs["tag"] = TAGS[kind]
        tr.attrs["color_override"] = (40, 40, 235)

        fine = getattr(self.cfg, kind)
        frame = frames.get(a)
        stem = f"{kind}_{a}_{tr.track_id}_{time.strftime('%H%M%S')}_{next(_seq)}"
        snap = close = None
        if frame is not None:
            img = EvidenceStore.mark(frame, [tr.box], f"{TAGS[kind]} #{tr.track_id}",
                                     f"{self.app.junction.name} | {a} approach | "
                                     f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {tr.speed_kmph:.0f} km/h")
            snap = self.evidence.save_image(img, stem)
            crop = EvidenceStore.crop(frame, tr.box)
            if crop is not None:
                close = self.evidence.save_image(crop, stem + "_crop")
        title = VIOLATION_TITLES[kind]
        info = {
            "violation": kind, "approach": a, "track_id": tr.track_id, "vehicle": tr.cls,
            "plate": plate, "speed_kmph": round(tr.speed_kmph, 1), "t": round(t, 1),
            "amount": fine.amount, "section": fine.section, "details": details,
        }
        self.counts[kind] += 1
        self.bus.publish(Event(EventType.VIOLATION,
                               f"{title} on {a}: {tr.cls} #{tr.track_id}" + (f" ({plate})" if plate else ""),
                               approach=a, severity=Severity.WARN, data=dict(info),
                               evidence=[p for p in (snap, close) if p]))
        crop_img = EvidenceStore.crop(frame, tr.box, pad=0.1) if (frame is not None and not plate) else None
        self.evidence.submit(lambda: self._issue(info, snap, close, crop_img))

    def _issue(self, info: dict, snap: Optional[str], close: Optional[str], crop_img) -> None:
        """Evidence worker thread: read plate (video), make the PDF, publish FINE_ISSUED."""
        plate = info["plate"]
        plate_info = None
        if plate is None and self.plates is not None and crop_img is not None:
            plate_info = self.plates.read(crop_img)
            plate = plate_info.get("plate")
            if plate:
                pkey = (plate, info["violation"])
                last = self._recent_plate.get(pkey)
                if last is not None and info["t"] - last < self.cfg.dedup_s:
                    return
                self._recent_plate[pkey] = info["t"]
        status = "ISSUED" if plate else "REVIEW"
        ts = time.time()
        no = "TA" + time.strftime("%y%m%d%H%M%S", time.localtime(ts)) + f"{next(_seq) % 100:02d}"
        loc = self.app.junction.location
        pdf_info = dict(info, challan_no=no, ts=ts, plate=plate, plate_status=status,
                        junction=self.app.junction.name, lat=loc.lat, lon=loc.lon)
        full, rel = self.evidence.new_path(f"challan_{no}", "pdf")
        make_challan_pdf(full, pdf_info, self.evidence.abs(snap) if snap else None,
                         self.evidence.abs(close) if close else None)
        data = {k: v for k, v in pdf_info.items() if k not in ("details",)}
        data["details"] = info["details"]
        if plate_info:
            data["anpr"] = plate_info
        self.fines_total += info["amount"]
        row = {"challan_no": no, "plate": plate, "violation": info["violation"], "approach": info["approach"],
               "amount": info["amount"], "status": status, "pdf": rel, "snapshot": snap, "ts": ts}
        self.recent = ([row] + self.recent)[:20]
        msg = (f"e-Challan {no}: Rs.{info['amount']} to {plate}" if plate
               else f"e-Challan {no}: Rs.{info['amount']}, plate not read, sent for manual review")
        self.bus.publish(Event(EventType.FINE_ISSUED, msg, approach=info["approach"],
                               severity=Severity.INFO, data=data, evidence=[rel] + ([snap] if snap else [])))

    def status(self) -> dict:
        return {"enabled": self.cfg.enabled, "counts": dict(self.counts),
                "total": sum(self.counts.values()), "fines_rs": self.fines_total,
                "recent": self.recent[:8], "rules": self.capabilities()}


def make_plate_reader(cfg: AppConfig):
    """ANPR for video mode; None (with a warning) if the models are missing."""
    if cfg.mode == "sim" or not cfg.violations.read_plates:
        return None
    try:
        from ..perception.plates import PlateReader
        return PlateReader(resolve(cfg.violations.plate_detector), resolve(cfg.violations.plate_ocr))
    except Exception as exc:
        print(f"[violations] ANPR disabled: {exc}")
        return None
