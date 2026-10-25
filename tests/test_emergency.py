from traffic_agent.config import AppConfig, EmergencyCfg, SignalCfg
from traffic_agent.control.adaptive import AdaptiveController
from traffic_agent.control.emergency import EmergencyModule
from traffic_agent.core.events import EventBus
from traffic_agent.core.pipeline import Pipeline
from traffic_agent.core.types import ApproachState, Track, WorldState

import numpy as np


def ambulance_delay(enabled: bool, approach="W", seed=11) -> float:
    """Seconds from injecting an ambulance (behind a queue) until it crosses the stop line."""
    cfg = AppConfig()
    cfg.sim.realtime = False
    cfg.sim.seed = seed
    cfg.emergency.enabled = enabled
    p = Pipeline(cfg, EventBus())
    dt = 1.0 / cfg.sim.fps
    for _ in range(int(90 / dt)):            # let queues build up
        p.tick()
    amb = p.sim.world.inject(approach, "ambulance", siren=True)
    t0 = p.now()
    while not amb.crossed_stop and p.now() - t0 < 300:
        p.tick()
    return p.now() - t0


def test_priority_gets_ambulance_through_faster():
    # W is the quiet approach: without priority it sits through other lanes' greens.
    with_prio, without = ambulance_delay(True, "W"), ambulance_delay(False, "W")
    assert with_prio < 20 and without > 3 * with_prio, (with_prio, without)
    with_prio, without = ambulance_delay(True, "S"), ambulance_delay(False, "S")
    assert with_prio < 25 and with_prio < without, (with_prio, without)


def _world(t, tracks_by_app):
    aps = {a: ApproachState(name=a, t=t, tracks=tracks_by_app.get(a, []), count_in_roi=len(tracks_by_app.get(a, [])))
           for a in "NESW"}
    return WorldState(t=t, approaches=aps, signal={})


def _amb(tid=1, inbound=True):
    return Track(track_id=tid, box=np.zeros(4), cls="car", score=0.9, attrs={"emergency": True},
                 confirmed=True, in_roi=inbound)


def test_preempts_current_green_and_releases_after_passing():
    ctl = AdaptiveController(SignalCfg(controller="adaptive"))
    em = EmergencyModule(EmergencyCfg(), EventBus(), ctl)
    ctl.current, ctl.stage, ctl.stage_end = "N", "GREEN", 40.0
    t = 1.0
    w = _world(t, {"E": [_amb()]})
    em.update(w, {}); ctl.update(w)
    assert ctl.override == "E" and ctl.stage == "AMBER"            # N cut short, safely
    while ctl.stage != "GREEN":
        t += 0.1; w = _world(t, {"E": [_amb()]}); em.update(w, {}); ctl.update(w)
    assert ctl.current == "E"
    for _ in range(300):                                           # held for 30 s while it's there
        t += 0.1; w = _world(t, {"E": [_amb()]}); em.update(w, {}); ctl.update(w)
    assert ctl.states()["E"] == "GREEN"
    for _ in range(30):                                            # gone for 3 s -> released
        t += 0.1; w = _world(t, {}); em.update(w, {}); ctl.update(w)
    assert ctl.override is None and em.served[0]["why"] == "vehicle has passed"


def test_false_detection_cannot_hold_green_forever():
    ctl = AdaptiveController(SignalCfg(controller="adaptive"))
    em = EmergencyModule(EmergencyCfg(max_hold_s=60), EventBus(), ctl)
    t = 0.0
    for _ in range(700):
        t += 0.1; w = _world(t, {"S": [_amb()]}); em.update(w, {}); ctl.update(w)
    assert ctl.override is None and "timed out" in em.served[0]["why"]


def test_manual_request():
    ctl = AdaptiveController(SignalCfg(controller="adaptive"))
    em = EmergencyModule(EmergencyCfg(manual_hold_s=10), EventBus(), ctl)
    em.request_manual("W")
    em.update(_world(0.0, {}), {})
    assert ctl.override == "W"
    em.update(_world(11.0, {}), {})
    assert ctl.override is None


def test_ambulance_driving_away_is_ignored():
    ctl = AdaptiveController(SignalCfg(controller="adaptive"))
    em = EmergencyModule(EmergencyCfg(), EventBus(), ctl)
    em.update(_world(1.0, {"E": [_amb(inbound=False)]}), {})       # leaving on E's outbound lane
    assert ctl.override is None and not em.requests


def test_one_ambulance_one_request_end_to_end():
    cfg = AppConfig()
    cfg.sim.realtime = False
    p = Pipeline(cfg, EventBus())
    for _ in range(300):
        p.tick()
    p.sim.world.inject("W", "ambulance", siren=True)
    for _ in range(600):                                           # 60 s: crosses and drives away
        p.tick()
    assert [r["approach"] for r in p.emergency.served] == ["W"]


def test_random_ambulances_arrive_at_configured_rate():
    from traffic_agent.sim.world import JunctionWorld
    w = JunctionWorld(arrivals_per_min={a: 0 for a in "NESW"}, emergency_per_hour=6, seed=3)
    seen = set()
    for _ in range(5 * 3600):                        # 5 simulated hours
        w.step(1.0)
        seen |= {v.vid for v in w.vehicles if v.kind == "ambulance" and v.siren}
    assert 15 <= len(seen) <= 48                     # ~30 expected


def test_random_ambulances_use_their_own_random_stream():
    from traffic_agent.sim.world import JunctionWorld
    def normal(rate):
        w = JunctionWorld(emergency_per_hour=rate, seed=7)
        for _ in range(600):
            w.step(0.1)
        return [v.vid for v in w.vehicles if v.kind != "ambulance"][:5], w.rng.random()
    assert normal(0)[1] == normal(500)[1]
