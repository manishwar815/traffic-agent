"""Zones: where the lanes, stop line and scale are inside one camera image.

Saved as JSON per camera (made with scripts/draw_zones.py, or generated
automatically for the simulator). Schema:

{
  "px_per_m": 12.0,                    # rough scale (used if no "calib")
  "anchor": "bottom",                  # "bottom" for normal cameras, "center" for top-down
  "lanes": [
    {"name": "inbound",  "inbound": true,  "polygon": [[x,y],...], "direction": [[x1,y1],[x2,y2]]},
    {"name": "outbound", "inbound": false, "polygon": [[x,y],...], "direction": [[x1,y1],[x2,y2]]}
  ],
  "stop_line": [[x1,y1],[x2,y2]],      # vehicles cross it in the inbound direction
  "calib": {"image": [[x,y]x4], "world": [[X,Y]x4]},   # optional, metres -> better speed
  "light_roi": [x1,y1,x2,y2]           # optional, a real traffic light visible in video
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class Lane:
    name: str
    polygon: np.ndarray            # (N, 2) float32 pixels
    direction: np.ndarray          # unit vector in pixels: legal travel direction
    inbound: bool = True

    def contains(self, pt: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(pt[0]), float(pt[1])), False) >= 0


@dataclass
class Zones:
    lanes: list[Lane]
    stop_line: Optional[np.ndarray] = None     # (2, 2)
    px_per_m: float = 10.0
    anchor: str = "bottom"
    calib_h: Optional[np.ndarray] = None       # 3x3 homography image -> metres
    light_roi: Optional[list[int]] = None
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, d: dict) -> "Zones":
        lanes = []
        for ln in d.get("lanes", []):
            p = np.array(ln["direction"], np.float32)
            v = p[1] - p[0]
            n = float(np.linalg.norm(v)) or 1.0
            lanes.append(Lane(ln.get("name", "lane"), np.array(ln["polygon"], np.float32), v / n,
                              bool(ln.get("inbound", True))))
        h = None
        if d.get("calib"):
            h = cv2.getPerspectiveTransform(np.array(d["calib"]["image"], np.float32),
                                            np.array(d["calib"]["world"], np.float32))
        sl = np.array(d["stop_line"], np.float32) if d.get("stop_line") else None
        return cls(lanes, sl, float(d.get("px_per_m", 10.0)), d.get("anchor", "bottom"), h,
                   d.get("light_roi"), d)

    @classmethod
    def load(cls, path: str | Path) -> "Zones":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def full_frame(cls, w: int, h: int) -> "Zones":
        """Fallback when no zones file exists: whole image is one inbound lane
        pointing up the image. Counting works; direction/stop-line rules are
        meaningless until you draw real zones (so "fallback" switches them off)."""
        return cls.from_dict({"px_per_m": 10.0, "fallback": True, "lanes": [{
            "name": "inbound", "inbound": True,
            "polygon": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            "direction": [[w / 2, h - 1], [w / 2, 0]]}]})

    # ------------------------------------------------------------- geometry
    @property
    def inbound_lanes(self) -> list[Lane]:
        return [ln for ln in self.lanes if ln.inbound]

    def lane_at(self, pt) -> Optional[Lane]:
        for ln in self.lanes:
            if ln.contains(pt):
                return ln
        return None

    def to_metres(self, pt: tuple[float, float]) -> tuple[float, float]:
        if self.calib_h is not None:
            p = cv2.perspectiveTransform(np.array([[pt]], np.float32), self.calib_h)[0, 0]
            return float(p[0]), float(p[1])
        return pt[0] / self.px_per_m, pt[1] / self.px_per_m

    def stop_line_side(self, pt) -> Optional[float]:
        """Signed distance (px) of pt from the stop line.
        Negative = before the line, positive = past it (in the inbound direction)."""
        if self.stop_line is None or not self.inbound_lanes:
            return None
        a, b = self.stop_line
        line = b - a
        normal = np.array([-line[1], line[0]], np.float32)
        normal /= float(np.linalg.norm(normal)) or 1.0
        if float(np.dot(normal, self.inbound_lanes[0].direction)) < 0:
            normal = -normal       # make "positive" point the way traffic flows
        return float(np.dot(np.array(pt, np.float32) - a, normal))

    def anchor_point(self, box) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        if self.anchor == "center":
            return (x1 + x2) / 2, (y1 + y2) / 2
        return (x1 + x2) / 2, y2

    def roi_mask(self, w: int, h: int) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        for ln in self.inbound_lanes:
            cv2.fillPoly(m, [ln.polygon.astype(np.int32)], 1)
        return m
