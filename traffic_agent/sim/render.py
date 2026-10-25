"""Draws the simulator as camera images.

* One "pole camera" per approach, looking down at that arm: inbound traffic
  moves left -> right toward the junction (upper lane); outbound traffic moves
  right -> left (lower lane). The junction box is visible on the right.
* One top-down overview of the whole junction (for the dashboard).

Rendering is plain OpenCV drawing. A static background is drawn once and
copied each frame, so it is cheap.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from .world import APPROACH_DIR, BOX_HALF, LANE_W, STOP_GAP, JunctionWorld, SimVehicle, left_of

GRASS = (62, 88, 66)
FOOTPATH = (150, 150, 145)
ROAD = (58, 58, 58)
WHITE = (235, 235, 235)
YELLOW = (60, 200, 230)
LIGHT_BGR = {"RED": (40, 40, 230), "AMBER": (30, 170, 250), "GREEN": (60, 200, 60)}
ROAD_HALF = LANE_W          # half width of a two-lane road


class Projection:
    """Maps world metres (x east, y north) to image pixels for one camera."""

    def __init__(self, to_px: Callable[[float, float], tuple[float, float]], ppm: float,
                 rot: Callable[[tuple[float, float]], tuple[float, float]]) -> None:
        self.to_px = to_px
        self.ppm = ppm
        self.rot = rot            # world direction -> image direction

    def pts(self, xy_list) -> np.ndarray:
        return np.array([self.to_px(x, y) for x, y in xy_list], dtype=np.float32)


def approach_projection(approach: str, width: int, height: int, ppm: float) -> Projection:
    d = APPROACH_DIR[approach]
    lft = left_of(d)
    a_max = BOX_HALF + 4.0                 # show a little beyond the junction box
    a_min = a_max - width / ppm

    def to_px(x: float, y: float) -> tuple[float, float]:
        a = x * d[0] + y * d[1]
        b = x * lft[0] + y * lft[1]
        return (a - a_min) * ppm, height / 2 - b * ppm

    def rot(v: tuple[float, float]) -> tuple[float, float]:
        return (v[0] * d[0] + v[1] * d[1], -(v[0] * lft[0] + v[1] * lft[1]))

    proj = Projection(to_px, ppm, rot)
    proj.a_min, proj.a_max = a_min, a_max   # type: ignore[attr-defined]
    return proj


def overview_projection(size: int, radius_m: float) -> Projection:
    ppm = size / (2 * radius_m)
    return Projection(lambda x, y: ((x + radius_m) * ppm, (radius_m - y) * ppm), ppm,
                      lambda v: (v[0], -v[1]))


# ---------------------------------------------------------------------- background
def _poly(img, proj: Projection, xy, color) -> None:
    cv2.fillPoly(img, [proj.pts(xy).astype(np.int32)], color, lineType=cv2.LINE_AA)


def draw_background(img: np.ndarray, proj: Projection, reach: float) -> None:
    img[:] = GRASS
    R, fp = reach, 2.0
    # footpaths then roads (N-S and E-W)
    _poly(img, proj, [(-ROAD_HALF - fp, -R), (ROAD_HALF + fp, -R), (ROAD_HALF + fp, R), (-ROAD_HALF - fp, R)], FOOTPATH)
    _poly(img, proj, [(-R, -ROAD_HALF - fp), (R, -ROAD_HALF - fp), (R, ROAD_HALF + fp), (-R, ROAD_HALF + fp)], FOOTPATH)
    _poly(img, proj, [(-ROAD_HALF, -R), (ROAD_HALF, -R), (ROAD_HALF, R), (-ROAD_HALF, R)], ROAD)
    _poly(img, proj, [(-R, -ROAD_HALF), (R, -ROAD_HALF), (R, ROAD_HALF), (-R, ROAD_HALF)], ROAD)
    for appr, d in APPROACH_DIR.items():
        lft = left_of(d)
        s_line = BOX_HALF + STOP_GAP    # distance of stop line from centre

        def P(a, b):  # point at a metres "before" centre along -d, b metres to the driver's left
            return (-d[0] * a + lft[0] * b, -d[1] * a + lft[1] * b)

        # dashed centre line
        a = s_line + 1.0
        while a < R:
            p1, p2 = proj.to_px(*P(a, 0)), proj.to_px(*P(min(a + 3, R), 0))
            cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), WHITE, max(1, int(proj.ppm * 0.15)))
            a += 6.0
        # zebra crossing between stop line and box
        for b in np.arange(-ROAD_HALF + 0.3, ROAD_HALF, 0.9):
            _poly(img, proj, [P(BOX_HALF + 0.4, b), P(s_line - 0.4, b), P(s_line - 0.4, b + 0.45),
                              P(BOX_HALF + 0.4, b + 0.45)], WHITE)
        # stop line across the inbound lane only
        _poly(img, proj, [P(s_line, 0), P(s_line + 0.4, 0), P(s_line + 0.4, ROAD_HALF), P(s_line, ROAD_HALF)], WHITE)
    # yellow box-junction markings
    c = [proj.to_px(x, y) for x, y in [(-BOX_HALF, -BOX_HALF), (BOX_HALF, -BOX_HALF),
                                        (BOX_HALF, BOX_HALF), (-BOX_HALF, BOX_HALF)]]
    cv2.polylines(img, [np.array(c, np.int32)], True, YELLOW, max(1, int(proj.ppm * 0.12)), cv2.LINE_AA)


# ---------------------------------------------------------------------- vehicles
def _rot_rect(center, size, angle_deg) -> np.ndarray:
    return cv2.boxPoints((center, size, angle_deg)).astype(np.int32)


def draw_vehicle(img: np.ndarray, proj: Projection, v: SimVehicle, arm_len: float, t: float) -> None:
    fx, fy = v.front_xy(arm_len)
    d = v.dir
    if v.wrong_way:
        d = (d[0], d[1])
    cx, cy = fx - d[0] * v.length / 2, fy - d[1] * v.length / 2
    u, w = proj.to_px(cx, cy)
    hx, hy = proj.rot(d)                       # heading in image coordinates
    ang = math.degrees(math.atan2(hy, hx))
    if v.crashed:
        ang += 18 if v.vid % 2 else -22        # visibly skewed after impact
    L, W = v.length * proj.ppm, v.width * proj.ppm
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))

    def local(along: float, across: float) -> tuple[float, float]:
        """Point in vehicle frame (along: +front, across: +right) -> image."""
        return u + ca * along * L - sa * across * W, w + sa * along * L + ca * across * W

    body = _rot_rect((u, w), (L, W), ang)
    if v.kind == "motorcycle":
        cv2.fillPoly(img, [_rot_rect((u, w), (L, W * 0.55), ang)], (40, 40, 40), cv2.LINE_AA)
        hx_, hy_ = local(-0.05, 0)
        r = max(2, int(W * 0.42))
        if v.helmet:
            cv2.circle(img, (int(hx_), int(hy_)), r, v.color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(hx_), int(hy_)), max(1, r // 2), (25, 25, 25), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (int(hx_), int(hy_)), r, (120, 160, 205), -1, cv2.LINE_AA)   # skin
            px, py = local(-0.12, 0)
            cv2.circle(img, (int(px), int(py)), max(1, int(r * 0.7)), (20, 20, 30), -1, cv2.LINE_AA)  # hair
    else:
        cv2.fillPoly(img, [body], v.color, cv2.LINE_AA)
        cv2.polylines(img, [body], True, tuple(int(c * 0.55) for c in v.color), 1, cv2.LINE_AA)
        # windshield band near the front
        ws = np.array([local(0.30, -0.42), local(0.18, -0.42), local(0.18, 0.42), local(0.30, 0.42)], np.int32)
        cv2.fillPoly(img, [ws], (70, 55, 45), cv2.LINE_AA)
        if v.kind == "bus":
            for k in (-0.15, 0.05):
                r_ = np.array([local(k, -0.3), local(k - 0.12, -0.3), local(k - 0.12, 0.3), local(k, 0.3)], np.int32)
                cv2.fillPoly(img, [r_], tuple(int(c * 0.8) for c in v.color), cv2.LINE_AA)
        if v.kind == "truck":
            cut = np.array([local(0.12, -0.5), local(0.08, -0.5), local(0.08, 0.5), local(0.12, 0.5)], np.int32)
            cv2.fillPoly(img, [cut], (30, 30, 30), cv2.LINE_AA)
        if v.kind == "ambulance":
            for side in (-0.45, 0.45):        # red side stripes
                st = np.array([local(0.15, side - 0.04), local(-0.48, side - 0.04),
                               local(-0.48, side + 0.04), local(0.15, side + 0.04)], np.int32)
                cv2.fillPoly(img, [st], (40, 40, 210), cv2.LINE_AA)
            cxp, cyp = local(-0.15, 0)        # red cross on the roof
            k = max(2, int(W * 0.18))
            cv2.line(img, (int(cxp - k), int(cyp)), (int(cxp + k), int(cyp)), (40, 40, 210), max(1, k // 2))
            cv2.line(img, (int(cxp), int(cyp - k)), (int(cxp), int(cyp + k)), (40, 40, 210), max(1, k // 2))
            if v.siren:                       # light bar flashing red / blue at 2 Hz
                on = int(t * 4) % 2 == 0
                for side, col in ((-0.25, (40, 40, 255)), (0.25, (255, 80, 30))):
                    lx, ly = local(0.12, side)
                    lit = col if (on == (side < 0)) else tuple(int(c * 0.3) for c in col)
                    cv2.circle(img, (int(lx), int(ly)), max(2, int(W * 0.16)), lit, -1, cv2.LINE_AA)
    if v.crashed:                              # debris / smoke marks
        rng = np.random.default_rng(v.vid)
        for _ in range(6):
            ox, oy = rng.normal(0, L * 0.35, 2)
            cv2.circle(img, (int(u + ox), int(w + oy)), max(1, int(proj.ppm * 0.2)), (20, 120, 230), -1)


def draw_signal_head(img: np.ndarray, center: tuple[int, int], state: str, scale: float = 1.0) -> None:
    r = int(7 * scale)
    x, y = center
    cv2.rectangle(img, (x - r - 3, y - 3 * r - 6), (x + r + 3, y + 3 * r + 6), (25, 25, 25), -1)
    for i, name in enumerate(("RED", "AMBER", "GREEN")):
        col = LIGHT_BGR[name] if state == name else (55, 55, 55)
        cv2.circle(img, (x, y - 2 * r - 2 + i * (2 * r + 2)), r, col, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------- renderer
class SimRenderer:
    def __init__(self, world: JunctionWorld, width: int, height: int, ppm: float,
                 overview_size: int = 640) -> None:
        self.world = world
        self.size = (width, height)
        self.proj = {a: approach_projection(a, width, height, ppm) for a in APPROACH_DIR}
        self.bg = {}
        for a, p in self.proj.items():
            img = np.zeros((height, width, 3), np.uint8)
            draw_background(img, p, reach=world.arm_len)
            self.bg[a] = img
        self.ov_proj = overview_projection(overview_size, radius_m=45.0)
        self.ov_bg = np.zeros((overview_size, overview_size, 3), np.uint8)
        draw_background(self.ov_bg, self.ov_proj, reach=world.arm_len)

    def render_approach(self, approach: str) -> np.ndarray:
        img = self.bg[approach].copy()
        p = self.proj[approach]
        for v in self.world.vehicles:
            draw_vehicle(img, p, v, self.world.arm_len, self.world.t)
        # this approach's signal head, mounted just past the stop line
        sx, _ = p.to_px(*self._stop_point(approach))
        draw_signal_head(img, (int(sx) + 18, 40), self.world.signal[approach])
        return img

    def render_overview(self) -> np.ndarray:
        img = self.ov_bg.copy()
        for v in self.world.vehicles:
            draw_vehicle(img, self.ov_proj, v, self.world.arm_len, self.world.t)
        for a, d in APPROACH_DIR.items():
            lft = left_of(d)
            s = BOX_HALF + STOP_GAP + 3
            x, y = -d[0] * s + lft[0] * (ROAD_HALF + 4), -d[1] * s + lft[1] * (ROAD_HALF + 4)
            u, w = self.ov_proj.to_px(x, y)
            draw_signal_head(img, (int(u), int(w)), self.world.signal[a], scale=0.7)
            lx, ly = self.ov_proj.to_px(-d[0] * 40, -d[1] * 40)
            cv2.putText(img, a, (int(lx) - 8, int(ly) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        return img

    def _stop_point(self, approach: str) -> tuple[float, float]:
        d = APPROACH_DIR[approach]
        s = BOX_HALF + STOP_GAP
        return -d[0] * s, -d[1] * s
