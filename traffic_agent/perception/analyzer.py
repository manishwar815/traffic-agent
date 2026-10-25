"""Turns raw tracks + zones into traffic measurements for one approach.

Per track: ground position in metres, speed, heading check, lane, stop-line crossing.
Per approach: vehicles in lane, queue length, occupancy, flow per minute.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..config import AnalysisCfg
from ..core.types import ApproachState, Track
from .zones import Zones


class ApproachAnalyzer:
    def __init__(self, name: str, zones: Zones, cfg: AnalysisCfg, frame_wh: tuple[int, int]) -> None:
        self.name = name
        self.zones = zones
        self.cfg = cfg
        self.mask = zones.roi_mask(*frame_wh)
        self.roi_area = max(1, int(self.mask.sum()))
        self.crossings: deque[float] = deque()        # times of stop-line crossings
        self._last_side: dict[int, float] = {}
        self._px_hist: dict[int, deque] = {}          # pixel positions per track (for heading)

    def update(self, tracks: list[Track], t: float) -> ApproachState:
        z = self.zones
        st = ApproachState(name=self.name, t=t, tracks=tracks)
        occ_px = 0
        for tr in tracks:
            px = z.anchor_point(tr.box)
            xm, ym = z.to_metres(px)
            tr.history.append((t, xm, ym))
            while tr.history and t - tr.history[0][0] > 3.0:
                tr.history.pop(0)
            ph = self._px_hist.setdefault(tr.track_id, deque())
            ph.append((t, px[0], px[1]))
            while ph and t - ph[0][0] > 3.0:
                ph.popleft()
            self._speed_and_heading(tr, px, ph)
            lane = z.lane_at(px)
            tr.in_roi = bool(lane and lane.inbound)
            tr.attrs["lane"] = lane.name if lane else None
            # stop-line crossing: sign of the signed distance flips - -> +
            tr.crossed_stop_line = False
            side = z.stop_line_side(px)
            if side is not None:
                prev = self._last_side.get(tr.track_id)
                lane_ok = tr.in_roi or tr.attrs.get("was_inbound")
                if prev is not None and prev < 0 <= side and lane_ok and tr.crossed_at is None:
                    tr.crossed_stop_line = True
                    tr.crossed_at = t
                    self.crossings.append(t)
                self._last_side[tr.track_id] = side
            if tr.in_roi:
                tr.attrs["was_inbound"] = True
                st.count_in_roi += 1
                if len(tr.history) >= 3 and tr.speed_kmph < self.cfg.queue_speed_kmph:
                    st.queue_len += 1
                x1, y1, x2, y2 = [int(v) for v in tr.box]
                occ_px += int(self.mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].sum())
        while self.crossings and t - self.crossings[0] > 60:
            self.crossings.popleft()
        st.flow_per_min = float(len(self.crossings))
        st.occupancy = min(1.0, occ_px / self.roi_area)
        alive = {tr.track_id for tr in tracks}
        if len(self._last_side) > 500:
            self._last_side = {k: v for k, v in self._last_side.items() if k in alive}
            self._px_hist = {k: v for k, v in self._px_hist.items() if k in alive}
        return st

    def _speed_and_heading(self, tr: Track, px, ph: deque) -> None:
        h = tr.history
        if len(h) < 2:
            return
        t_now = h[-1][0]
        old = next((p for p in h if t_now - p[0] <= self.cfg.speed_window_s), h[0])
        dt = t_now - old[0]
        if dt <= 0.2:
            return
        dist_m = math.hypot(h[-1][1] - old[1], h[-1][2] - old[2])
        v = dist_m / dt * 3.6
        tr.speed_kmph = 0.5 * tr.speed_kmph + 0.5 * v if tr.speed_kmph else v
        # Heading: compare pixel motion with the lane's legal direction (pixel vector).
        if dist_m > 1.0:
            p_old = next((p for p in ph if t_now - p[0] <= self.cfg.speed_window_s), ph[0])
            motion = np.array([ph[-1][1] - p_old[1], ph[-1][2] - p_old[2]], np.float32)
            n = float(np.linalg.norm(motion))
            lane = self.zones.lane_at(px)
            if lane is not None and n > 0:
                tr.heading_ok = float(np.dot(motion / n, lane.direction)) > 0
