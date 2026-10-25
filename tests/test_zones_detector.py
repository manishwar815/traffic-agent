from pathlib import Path

import numpy as np
import pytest

from traffic_agent.perception.detector import YoloOnnxDetector, letterbox
from traffic_agent.perception.zones import Zones

ROOT = Path(__file__).resolve().parent.parent

ZONES = {
    "px_per_m": 10,
    "lanes": [{"name": "inbound", "inbound": True, "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
               "direction": [[0, 25], [100, 25]]}],
    "stop_line": [[80, 0], [80, 50]],
}


def test_stop_line_sign_follows_traffic_direction():
    z = Zones.from_dict(ZONES)
    assert z.stop_line_side((70, 25)) < 0 < z.stop_line_side((90, 25))
    # same line drawn in the opposite order must give the same sign
    d = dict(ZONES, stop_line=[[80, 50], [80, 0]])
    z2 = Zones.from_dict(d)
    assert z2.stop_line_side((70, 25)) < 0 < z2.stop_line_side((90, 25))


def test_lane_lookup_and_metres():
    z = Zones.from_dict(ZONES)
    assert z.lane_at((10, 10)).name == "inbound"
    assert z.lane_at((10, 200)) is None
    assert z.to_metres((100, 50)) == (10.0, 5.0)


def test_letterbox_shape():
    img = np.zeros((360, 640, 3), np.uint8)
    out, r, dw, dh = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == 1.0 and dw == 0 and dh == 140


def test_raw_head_decoding_runs_nms():
    det = YoloOnnxDetector.__new__(YoloOnnxDetector)   # skip model loading
    det.conf, det.iou = 0.25, 0.5
    raw = np.zeros((84, 3), np.float32)
    raw[:4, 0] = [100, 100, 40, 20]; raw[4 + 2, 0] = 0.9   # car
    raw[:4, 1] = [102, 101, 40, 20]; raw[4 + 2, 1] = 0.8   # duplicate car -> removed by NMS
    raw[:4, 2] = [300, 300, 40, 20]; raw[4 + 7, 2] = 0.7   # truck
    rows = det._decode_raw(raw)
    assert len(rows) == 2 and set(rows[:, 5].astype(int)) == {2, 7}


@pytest.mark.skipif(not (ROOT / "models/yolo26n.onnx").exists(), reason="run scripts/download_models.py")
def test_real_model_loads_and_runs():
    det = YoloOnnxDetector(ROOT / "models/yolo26n.onnx")
    out = det(np.full((360, 640, 3), 90, np.uint8))
    assert isinstance(out, list)
