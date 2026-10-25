"""Simulator as a camera source + a 'simulated detector'.

Honest note: in sim mode the detector does NOT run YOLO. It reads vehicle
boxes straight from the simulator (with optional noise and misses), because
YOLO is trained on real photos and does not recognise drawn rectangles.
Everything AFTER detection (tracking, measurement, all four modules) is the
same code that runs on real video. Ground truth is stored under
attrs["gt"]; modules must not read it (tests use it to score accuracy).
"""
from __future__ import annotations

import random

import numpy as np

from ..config import SimCfg
from ..core.types import Detection
from ..perception.zones import Zones
from .render import ROAD_HALF, SimRenderer
from .world import APPROACH_DIR, BOX_HALF, STOP_GAP, JunctionWorld

# What a COCO-trained detector would call each simulated vehicle.
# (An ambulance is a van: COCO has no 'ambulance' class.)
SIM_TO_COCO = {"car": "car", "motorcycle": "motorcycle", "bus": "bus", "truck": "truck", "ambulance": "car"}


class SimJunctionSource:
    def __init__(self, cfg: SimCfg, noise_px: float = 1.0, miss_rate: float = 0.02) -> None:
        self.cfg = cfg
        self.world = JunctionWorld(arm_len=cfg.arm_length_m, arrivals_per_min=cfg.arrivals_per_min,
                                   vehicle_mix=cfg.vehicle_mix, speed_limit_kmph=cfg.speed_limit_kmph,
                                   seed=cfg.seed, emergency_per_hour=cfg.emergency_per_hour)
        self.renderer = SimRenderer(self.world, cfg.view_width, cfg.view_height, cfg.px_per_m)
        self.noise_px = noise_px
        self.miss_rate = miss_rate
        self.rng = random.Random(cfg.seed + 1)

    @property
    def t(self) -> float:
        return self.world.t

    def step(self, dt: float) -> list[dict]:
        return self.world.step(dt)

    def frame(self, approach: str) -> np.ndarray:
        return self.renderer.render_approach(approach)

    def overview(self) -> np.ndarray:
        return self.renderer.render_overview()

    # ---------------------------------------------------------------- detector
    def detections(self, approach: str) -> list[Detection]:
        proj = self.renderer.proj[approach]
        W, H = self.cfg.view_width, self.cfg.view_height
        dets = []
        for v in self.world.vehicles:
            x0, y0, x1, y1 = v.footprint(self.world.arm_len)
            pts = proj.pts([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            bx1, by1 = pts.min(0)
            bx2, by2 = pts.max(0)
            full = (bx2 - bx1) * (by2 - by1)
            cx1, cy1, cx2, cy2 = max(bx1, 0), max(by1, 0), min(bx2, W - 1), min(by2, H - 1)
            if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < 0.4 * full:
                continue                                   # mostly outside this camera
            if self.rng.random() < self.miss_rate:
                continue                                   # simulated missed detection
            box = np.array([cx1, cy1, cx2, cy2], np.float32)
            box += np.array([self.rng.gauss(0, self.noise_px) for _ in range(4)], np.float32)
            dets.append(Detection(
                box=box, score=self.rng.uniform(0.6, 0.95), cls=SIM_TO_COCO[v.kind],
                attrs={"gt": {"vid": v.vid, "kind": v.kind, "plate": v.plate, "helmet": v.helmet,
                              "siren": v.siren, "crashed": v.crashed, "speed_kmph": v.v * 3.6,
                              "wrong_way": v.wrong_way},
                       # Stand-ins for extra classifiers that run on real video:
                       # emergency-vehicle (siren/lightbar), helmet, and ANPR plate reading.
                       "emergency": v.kind == "ambulance" and v.siren,
                       "helmet": v.helmet,
                       "plate": v.plate}))
        return dets

    # ---------------------------------------------------------------- zones
    def zones(self, approach: str) -> Zones:
        """Generate the zones JSON for one sim camera from the known geometry."""
        p = self.renderer.proj[approach]
        H = self.cfg.view_height
        s_line = BOX_HALF + STOP_GAP
        x_stop, _ = p.to_px(*self._pt(approach, s_line, 0))
        y_mid = H / 2
        lane_px = ROAD_HALF * p.ppm
        x_end, _ = p.to_px(*self._pt(approach, BOX_HALF, 0))        # lane ends at the junction box
        return Zones.from_dict({
            "px_per_m": p.ppm,
            "metric": True,                                          # exact scale: speeds are real
            "anchor": "center",                                      # top-down camera
            "lanes": [
                {"name": "inbound", "inbound": True,
                 "polygon": [[0, y_mid - lane_px], [x_end, y_mid - lane_px], [x_end, y_mid], [0, y_mid]],
                 "direction": [[0, y_mid - lane_px / 2], [100, y_mid - lane_px / 2]]},
                {"name": "outbound", "inbound": False,
                 "polygon": [[0, y_mid], [x_end, y_mid], [x_end, y_mid + lane_px], [0, y_mid + lane_px]],
                 "direction": [[100, y_mid + lane_px / 2], [0, y_mid + lane_px / 2]]},
            ],
            "stop_line": [[x_stop, y_mid - lane_px], [x_stop, y_mid]],
        })

    @staticmethod
    def _pt(approach: str, a: float, b: float) -> tuple[float, float]:
        d = APPROACH_DIR[approach]
        return -d[0] * a + (-d[1]) * b, -d[1] * a + d[0] * b
