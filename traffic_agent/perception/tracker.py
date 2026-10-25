"""Multi-object tracker (simplified ByteTrack).

Gives every vehicle a stable ID across frames, which all modules need
(e.g. "vehicle #12 crossed the stop line while red").

ByteTrack's key idea, kept here: match confident detections first, then use
low-confidence detections only to *keep existing tracks alive* (a partly
hidden car often gets a low score; ignoring it would break its track).
Each track predicts its next box with a constant-velocity model, so fast
vehicles at low FPS still match.
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import TrackerCfg
from ..core.types import Detection, Track

BIG = 1e6


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


class _TrackState:
    """Internal bookkeeping wrapped around the public Track object."""

    def __init__(self, track: Track) -> None:
        self.track = track
        self.vel = np.zeros(4, np.float32)     # box change per frame
        self.cls_votes: dict[str, float] = {track.cls: track.score}

    def predicted(self) -> np.ndarray:
        return self.track.box + self.vel


class Tracker:
    def __init__(self, cfg: TrackerCfg, id_start: int = 1) -> None:
        self.cfg = cfg
        self._ids = itertools.count(id_start)
        self.states: list[_TrackState] = []

    # ------------------------------------------------------------------ matching
    def _cost(self, states: list[_TrackState], dets: list[Detection]) -> np.ndarray:
        pred = np.array([s.predicted() for s in states], np.float32).reshape(-1, 4)
        boxes = np.array([d.box for d in dets], np.float32).reshape(-1, 4)
        ious = iou_matrix(pred, boxes)
        cost = 1.0 - ious
        # Fallback for small fast objects whose boxes don't overlap: accept if the
        # centres are close relative to the object size.
        pc = (pred[:, None, :2] + pred[:, None, 2:]) / 2
        dc = (boxes[None, :, :2] + boxes[None, :, 2:]) / 2
        dist = np.linalg.norm(pc - dc, axis=2)
        size = np.maximum(pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1])[:, None]
        near = dist < 1.0 * size        # up to one body length per frame
        cost = np.where(ious >= self.cfg.match_iou, cost, np.where(near, 0.95, BIG))
        return cost

    def _match(self, states, dets):
        if not states or not dets:
            return [], list(range(len(states))), list(range(len(dets)))
        cost = self._cost(states, dets)
        rows, cols = linear_sum_assignment(cost)
        matches, mt, md = [], set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] < BIG:
                matches.append((r, c)); mt.add(r); md.add(c)
        return (matches, [i for i in range(len(states)) if i not in mt],
                [j for j in range(len(dets)) if j not in md])

    # ------------------------------------------------------------------ update
    def update(self, dets: Optional[list[Detection]]) -> list[Track]:
        """dets=None means 'no detection this frame' (frame skipping): tracks coast
        on their velocity without being counted as lost."""
        if dets is None:
            for s in self.states:
                s.track.box = s.predicted()
            return self.active()

        high = [d for d in dets if d.score >= self.cfg.high_thresh]
        low = [d for d in dets if self.cfg.low_thresh <= d.score < self.cfg.high_thresh]

        # 1) confident detections vs all tracks
        m1, un_t, un_d = self._match(self.states, high)
        for ti, di in m1:
            self._apply(self.states[ti], high[di])
        # 2) weak detections vs the tracks still unmatched
        rest = [self.states[i] for i in un_t]
        m2, un_t2, _ = self._match(rest, low)
        for ti, di in m2:
            self._apply(rest[ti], low[di])
        # 3) unmatched tracks coast and age
        for i in un_t2:
            s = rest[i]
            s.track.box = s.predicted()
            s.track.lost += 1
        # 4) new tracks from confident, unmatched detections
        for di in un_d:
            d = high[di]
            if d.score >= self.cfg.new_track_thresh:
                t = Track(track_id=next(self._ids), box=d.box.copy(), cls=d.cls, score=d.score,
                          attrs=dict(d.attrs))
                t.confirmed = self.cfg.min_hits <= 1
                self.states.append(_TrackState(t))
        # 5) drop long-lost tracks
        self.states = [s for s in self.states if s.track.lost <= self.cfg.max_lost]
        return self.active()

    def _apply(self, s: _TrackState, d: Detection) -> None:
        t = s.track
        frames = t.lost + 1
        s.vel = 0.6 * s.vel + 0.4 * (d.box - t.box) / frames
        t.box = d.box.copy()
        t.score = d.score
        t.hits += 1
        t.lost = 0
        t.attrs.update(d.attrs)
        s.cls_votes[d.cls] = s.cls_votes.get(d.cls, 0.0) + d.score
        t.cls = max(s.cls_votes, key=s.cls_votes.get)    # majority vote, stops label flicker
        if t.hits >= self.cfg.min_hits:
            t.confirmed = True

    def active(self) -> list[Track]:
        """Confirmed tracks seen in the current frame."""
        return [s.track for s in self.states if s.track.confirmed and s.track.lost == 0]
