"""Licence-plate reading (ANPR) for video mode, two small ONNX models on CPU:

  1. plate_detector.onnx  (open-image-models YOLOv9-t, 384x384, end-to-end NMS)
     input  images (1, 3, 384, 384) float RGB 0..1, letterboxed
     output (N, 7) = batch_idx, x1, y1, x2, y2, class, score
  2. plate_ocr.onnx       (fast-plate-ocr CCT-XS, global model)
     input  (N, 64, 128, 3) uint8 RGB, plain resize (no aspect ratio)
     output "plate" (N, 10, 37) per-slot character probabilities, '_' = padding

It only runs on violators (a few crops per minute), never on every frame.
In sim mode the simulator supplies the plate directly (see sim/source.py).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .detector import letterbox

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD = "_"
# Indian plate: state (2 letters), RTO (1-2 digits), series (0-3 letters), number (1-4 digits)
INDIAN_PLATE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$")


def decode_ocr(probs: np.ndarray, alphabet: str = ALPHABET, pad: str = PAD) -> tuple[str, float]:
    """(slots, len(alphabet)) probabilities -> (text, mean confidence of the kept characters)."""
    idx = probs.argmax(-1)
    conf = probs.max(-1)
    chars = [alphabet[i] for i in idx]
    keep = [k for k, c in enumerate(chars) if c != pad]
    text = "".join(chars[k] for k in keep)
    return text, float(conf[keep].mean()) if keep else 0.0


def decode_plate_boxes(out: np.ndarray, r: float, dw: float, dh: float, w: int, h: int,
                       min_score: float) -> list[tuple[np.ndarray, float]]:
    """Detector rows -> [(box in crop pixels, score)], best first."""
    found = []
    for row in out.reshape(-1, 7):
        score = float(row[6])
        if score < min_score:
            continue
        box = (row[1:5] - np.array([dw, dh, dw, dh], np.float32)) / r
        box = np.clip(box, 0, [w - 1, h - 1, w - 1, h - 1])
        if box[2] - box[0] >= 8 and box[3] - box[1] >= 4:
            found.append((box, score))
    return sorted(found, key=lambda x: -x[1])


class PlateReader:
    def __init__(self, detector_path: str | Path, ocr_path: str | Path, threads: int = 2,
                 det_size: int = 384, min_det_score: float = 0.4, min_ocr_conf: float = 0.5) -> None:
        for p in (detector_path, ocr_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"ANPR model not found: {p}\nRun:  python scripts/download_models.py")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.det = ort.InferenceSession(str(detector_path), opts, providers=["CPUExecutionProvider"])
        self.ocr = ort.InferenceSession(str(ocr_path), opts, providers=["CPUExecutionProvider"])
        self.det_in = self.det.get_inputs()[0].name
        self.ocr_in = self.ocr.get_inputs()[0].name
        self.det_size = det_size
        self.min_det_score = min_det_score
        self.min_ocr_conf = min_ocr_conf

    def find_plate(self, vehicle: np.ndarray) -> Optional[np.ndarray]:
        """Crop of the best plate inside a vehicle crop (BGR), or None."""
        img, r, dw, dh = letterbox(vehicle, self.det_size)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        out = self.det.run(None, {self.det_in: x})[0]
        h, w = vehicle.shape[:2]
        boxes = decode_plate_boxes(out, r, dw, dh, w, h, self.min_det_score)
        if not boxes:
            return None
        x1, y1, x2, y2 = boxes[0][0].astype(int)
        return vehicle[y1:y2 + 1, x1:x2 + 1]

    def read_plate_crop(self, plate: np.ndarray) -> tuple[str, float]:
        rgb = cv2.cvtColor(cv2.resize(plate, (128, 64), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        probs = self.ocr.run(None, {self.ocr_in: rgb[None]})[0][0]
        return decode_ocr(probs)

    def read(self, vehicle: np.ndarray) -> dict:
        """Vehicle crop (BGR) -> {"plate": text or None, "conf": float, "valid_format": bool}."""
        plate = self.find_plate(vehicle)
        if plate is None or plate.size == 0:
            return {"plate": None, "conf": 0.0, "valid_format": False, "why": "no plate found"}
        text, conf = self.read_plate_crop(plate)
        if conf < self.min_ocr_conf or len(text) < 4:
            return {"plate": None, "conf": conf, "valid_format": False, "why": f"unclear ({text})"}
        return {"plate": text, "conf": conf, "valid_format": bool(INDIAN_PLATE.match(text))}
