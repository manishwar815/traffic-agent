import numpy as np

from traffic_agent.config import TrackerCfg
from traffic_agent.core.types import Detection
from traffic_agent.perception.tracker import Tracker, iou_matrix


def det(x, y, w=40, h=20, score=0.9, cls="car"):
    return Detection(box=np.array([x, y, x + w, y + h], np.float32), score=score, cls=cls)


def test_iou_basic():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    m = iou_matrix(a, b)
    assert abs(m[0, 0] - 1 / 3) < 1e-3 and m[0, 1] == 0


def test_two_vehicles_keep_their_ids():
    tr = Tracker(TrackerCfg(min_hits=1))
    ids_a, ids_b = set(), set()
    for k in range(30):
        tracks = tr.update([det(10 + 8 * k, 50), det(400 - 8 * k, 120)])
        for t in tracks:
            (ids_a if t.box[1] < 100 else ids_b).add(t.track_id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_fast_small_object_without_overlap_still_matches():
    # 20 px wide bike moving 18 px per frame: consecutive boxes barely overlap
    tr = Tracker(TrackerCfg(min_hits=1))
    ids = {t.track_id for k in range(15) for t in tr.update([det(18 * k, 50, w=20, h=8)])}
    assert len(ids) == 1


def test_low_score_keeps_track_alive_but_does_not_create():
    tr = Tracker(TrackerCfg(min_hits=1))
    first = tr.update([det(100, 100)])[0].track_id
    for k in range(5):            # partially hidden: low score
        tracks = tr.update([det(100 + 2 * k, 100, score=0.2)])
        assert tracks and tracks[0].track_id == first
    tr2 = Tracker(TrackerCfg(min_hits=1))
    assert tr2.update([det(10, 10, score=0.2)]) == []


def test_lost_track_is_removed():
    cfg = TrackerCfg(min_hits=1, max_lost=3)
    tr = Tracker(cfg)
    tr.update([det(0, 0)])
    for _ in range(5):
        tr.update([])
    assert tr.states == []
