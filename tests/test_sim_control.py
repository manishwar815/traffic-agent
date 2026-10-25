from traffic_agent.config import SignalCfg, SimCfg
from traffic_agent.control.fixed import FixedTimeController
from traffic_agent.core.types import WorldState
from traffic_agent.sim.source import SimJunctionSource
from traffic_agent.sim.world import JunctionWorld


def run(world, seconds, dt=0.1):
    for _ in range(int(seconds / dt)):
        world.step(dt)


def test_red_light_stops_normal_traffic():
    w = JunctionWorld(arrivals_per_min={"N": 20, "E": 0, "S": 0, "W": 0}, seed=1)
    w.signal = {a: "RED" for a in "NESW"}
    run(w, 120)
    assert w.completed == []
    assert not any(v.crossed_stop for v in w.vehicles)
    w.signal["N"] = "GREEN"
    run(w, 60)
    assert len(w.completed) > 0


def test_red_light_jumper_crosses_on_red():
    w = JunctionWorld(arrivals_per_min={a: 0 for a in "NESW"}, seed=1)
    w.signal = {a: "RED" for a in "NESW"}
    v = w.inject("E", "car", ignore_red=True)
    run(w, 15)
    assert v.crossed_stop


def test_distracted_driver_rear_ends_queue():
    w = JunctionWorld(arrivals_per_min={a: 0 for a in "NESW"}, seed=3)
    w.signal = {a: "RED" for a in "NESW"}
    w.inject("S", "car")
    run(w, 20)                     # first car waits at the red light
    w.inject("S", "car", distracted=True)
    run(w, 20)
    assert len(w.crashes) == 1


def test_simulation_is_deterministic():
    def go():
        w = JunctionWorld(seed=7)
        w.signal["N"] = "GREEN"
        run(w, 60)
        return [(v.vid, round(v.s, 3)) for v in w.vehicles]
    assert go() == go()


def test_sim_detections_are_inside_frame():
    src = SimJunctionSource(SimCfg(seed=5))
    src.world.signal["W"] = "GREEN"
    for _ in range(300):
        src.step(0.1)
    for a in "NESW":
        for d in src.detections(a):
            x1, y1, x2, y2 = d.box
            assert -3 <= x1 < x2 <= 643 and -3 <= y1 < y2 <= 363
            assert "gt" in d.attrs


def test_fixed_controller_cycle():
    cfg = SignalCfg()
    c = FixedTimeController(cfg)
    seen, last = [], None
    for k in range(0, 1000):
        st = c.update(WorldState(t=k * 0.1, approaches={}, signal={}))
        g = [a for a, s in st.items() if s == "GREEN"]
        assert len(g) <= 1                       # never two greens at once
        if g and g[0] != last:
            seen.append(g[0])
        last = g[0] if g else last
    assert seen[:4] == ["N", "E", "S", "W"]
