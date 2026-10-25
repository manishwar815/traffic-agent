from traffic_agent.config import SignalCfg
from traffic_agent.control.adaptive import AdaptiveController
from traffic_agent.core.types import ApproachState, WorldState


def world(t, counts):
    aps = {a: ApproachState(name=a, t=t, count_in_roi=counts.get(a, 0)) for a in "NESW"}
    return WorldState(t=t, approaches=aps, signal={})


def test_busiest_lane_goes_first():
    c = AdaptiveController(SignalCfg(controller="adaptive"))
    c.update(world(0.0, {"N": 1, "E": 7, "S": 3, "W": 0}))
    assert c.current == "E" and c.stage == "GREEN"


def test_green_scales_with_queue_and_is_capped():
    c = AdaptiveController(SignalCfg(controller="adaptive"))
    assert c.green_time("N", world(0, {"N": 2})) == 8 + 2.5 * 2
    assert c.green_time("N", world(0, {"N": 100})) == 45


def test_empty_lane_gaps_out_after_min_green():
    c = AdaptiveController(SignalCfg(controller="adaptive"))
    c.update(world(0.0, {"N": 10}))            # N green for 33 s
    c.update(world(9.0, {"N": 0, "E": 2}))     # N emptied after min green
    assert c.stage == "AMBER"


def test_quiet_lane_is_not_starved():
    c = AdaptiveController(SignalCfg(controller="adaptive"))
    c.t, c.current = 100.0, "N"
    c.last_green = {"N": 100.0, "E": 95.0, "S": 0.0, "W": 90.0}
    assert c.choose_next(world(100.0, {"N": 9, "E": 9, "S": 1, "W": 0})) == "S"
