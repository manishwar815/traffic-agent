"""YOLO vehicle detector running on ONNX Runtime (CPU, no PyTorch).

Supports both Ultralytics ONNX output layouts:
  * YOLO26 (end-to-end, NMS-free): output (1, 300, 6) = x1, y1, x2, y2, score, class
  * YOLO11 / YOLOv8:               output (1, 84, 8400) = cx, cy, w, h, 80 class scores
    -> we run NMS ourselves.
Boxes are mapped back from the 640x640 letterboxed input to frame pixels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..core.types import COCO_TO_NAME, Detection


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping aspect ratio, pad to size x size with grey (as in training)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, dw, dh


class YoloOnnxDetector:
    def __init__(self, model_path: str | Path, input_size: int = 640, conf: float = 0.25,
                 classes: list[int] | None = None, threads: int = 4, iou: float = 0.5) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\nRun:  python scripts/download_models.py")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = input_size
        self.conf = conf
        self.iou = iou
        self.classes = set(classes) if classes else set(COCO_TO_NAME)
        self.name = model_path.stem

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        img, r, dw, dh = letterbox(frame, self.size)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: x})[0]
        if out.ndim == 3 and out.shape[2] == 6:
            rows = self._decode_end2end(out[0])
        else:
            rows = self._decode_raw(out[0])
        dets = []
        h, w = frame.shape[:2]
        for x1, y1, x2, y2, score, cls in rows:
            cls = int(cls)
            if cls not in self.classes or score < self.conf:
                continue
            box = (np.array([x1, y1, x2, y2], np.float32) - [dw, dh, dw, dh]) / r
            box = np.clip(box, 0, [w - 1, h - 1, w - 1, h - 1])
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            dets.append(Detection(box=box, score=float(score), cls=COCO_TO_NAME.get(cls, str(cls))))
        return dets

    def _decode_end2end(self, out: np.ndarray) -> np.ndarray:
        return out[out[:, 4] >= self.conf]

    def _decode_raw(self, out: np.ndarray) -> np.ndarray:
        """(84, N) raw head -> NMS'd rows of x1,y1,x2,y2,score,cls."""
        if out.shape[0] == 84 or (out.shape[0] < out.shape[1] and out.shape[1] != 84):
            out = out.T                                   # (84, N) -> (N, 84) for COCO models
        scores_all = out[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        keep = scores >= self.conf
        b, scores, cls = out[keep, :4], scores[keep], cls[keep]
        if not len(b):
            return np.zeros((0, 6), np.float32)
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        # class-aware NMS by offsetting boxes per class
        offs = xyxy + cls[:, None] * 4096.0
        xywh = np.stack([offs[:, 0], offs[:, 1], offs[:, 2] - offs[:, 0], offs[:, 3] - offs[:, 1]], 1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        return np.concatenate([xyxy[idx], scores[idx, None], cls[idx, None]], 1)
