"""Draws boxes, IDs, lanes and status onto frames for the dashboard."""
from __future__ import annotations

import cv2
import numpy as np

from ..perception.zones import Zones
from .types import ApproachState

CLS_COLOR = {"car": (255, 200, 60), "motorcycle": (80, 220, 255), "bus": (120, 255, 120),
             "truck": (255, 140, 200), "bicycle": (200, 200, 255), "ambulance": (60, 60, 255)}
SIG_COLOR = {"RED": (50, 50, 235), "AMBER": (30, 175, 255), "GREEN": (70, 210, 70)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_zones(img: np.ndarray, zones: Zones, signal: str) -> None:
    overlay = img.copy()
    for ln in zones.lanes:
        col = (200, 160, 40) if ln.inbound else (120, 120, 120)
        cv2.fillPoly(overlay, [ln.polygon.astype(np.int32)], col)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
    for ln in zones.lanes:
        col = (220, 180, 60) if ln.inbound else (150, 150, 150)
        cv2.polylines(img, [ln.polygon.astype(np.int32)], True, col, 1, cv2.LINE_AA)
    if zones.stop_line is not None:
        a, b = zones.stop_line.astype(int)
        cv2.line(img, tuple(a), tuple(b), SIG_COLOR.get(signal, (255, 255, 255)), 3, cv2.LINE_AA)


def draw_tracks(img: np.ndarray, st: ApproachState) -> None:
    for tr in st.tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        col = tr.attrs.get("color_override") or CLS_COLOR.get(tr.cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if tr.in_roi else 1, cv2.LINE_AA)
        label = f"#{tr.track_id} {tr.cls} {tr.speed_kmph:.0f}km/h"
        tag = tr.attrs.get("tag")             # set by modules, e.g. "AMBULANCE", "RED-LIGHT"
        if tag:
            label = f"{tag} " + label
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.38, 1)
        ty = y1 - 3 if y1 - th - 4 > 0 else y2 + th + 3
        cv2.rectangle(img, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), (20, 20, 20), -1)
        cv2.putText(img, label, (x1 + 1, ty), FONT, 0.38, col, 1, cv2.LINE_AA)


def draw_header(img: np.ndarray, st: ApproachState, extra: str = "") -> None:
    h = 22
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (20, 20, 20), -1)
    cv2.circle(img, (12, h // 2), 6, SIG_COLOR.get(st.signal, (200, 200, 200)), -1, cv2.LINE_AA)
    txt = (f"{st.name}  |  in lane {st.count_in_roi}  queue {st.queue_len}  "
           f"occ {st.occupancy * 100:.0f}%  flow {st.flow_per_min:.0f}/min  {st.fps:.1f} fps {extra}")
    cv2.putText(img, txt, (24, 15), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)


def annotate(frame: np.ndarray, zones: Zones, st: ApproachState, extra: str = "") -> np.ndarray:
    img = frame.copy()
    draw_zones(img, zones, st.signal)
    draw_tracks(img, st)
    draw_header(img, st, extra)
    return img
