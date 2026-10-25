"""Reads a REAL traffic light visible in a video (zones "light_roi").

Why: in video mode our own signal is virtual (recorded drivers never saw it),
so a red-light violation can only be judged against the light the driver
actually saw. Draw a box around the signal head with scripts/draw_zones.py.

Method: count bright, saturated red vs green pixels in the box (HSV).
Amber counts as "AMBER" (not a violation). Unknown = no decision.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def classify_light(crop: np.ndarray, min_frac: float = 0.02) -> Optional[str]:
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 90) & (v > 150)
    red = lit & ((h <= 8) | (h >= 170))
    amber = lit & (h > 12) & (h <= 30)
    green = lit & (h >= 45) & (h <= 95)
    n = float(crop.shape[0] * crop.shape[1])
    fr = {"RED": red.sum() / n, "AMBER": amber.sum() / n, "GREEN": green.sum() / n}
    best = max(fr, key=fr.get)
    return best if fr[best] >= min_frac else None


class LightReader:
    """Smooths per-frame readings: a state must hold for `stable_frames` frames."""

    def __init__(self, roi: list[int], stable_frames: int = 3) -> None:
        self.roi = [int(v) for v in roi]
        self.stable_frames = stable_frames
        self.state: Optional[str] = None
        self._cand: Optional[str] = None
        self._n = 0

    def update(self, frame: np.ndarray) -> Optional[str]:
        x1, y1, x2, y2 = self.roi
        s = classify_light(frame[max(0, y1):y2, max(0, x1):x2])
        if s == self._cand:
            self._n += 1
        else:
            self._cand, self._n = s, 1
        if self._n >= self.stable_frames:
            self.state = self._cand
        return self.state
