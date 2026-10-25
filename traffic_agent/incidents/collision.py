"""Collision detection + automatic ambulance alert.

A crash, as seen by a camera: two vehicles whose boxes touch, at least one
of which stopped SUDDENLY (lost min_drop_kmph within drop_window_s), and
both stay stationary and touching for confirm_s. Normal queues do not
trigger it: queued vehicles keep a gap and slow down gradually.

Then:
  COLLISION event (critical) with a snapshot of both vehicles
  -> alert on the evidence worker (Telegram, or a dry-run record)
  -> AMBULANCE_CALLED event with the delivery result.
The same crash seen from another camera within merge_s is added to the same
incident instead of calling a second ambulance.
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from typing import Optional

import numpy as np

from ..config import AppConfig
from ..core.events import Event, EventBus, EventType, Severity
from ..core.evidence import EvidenceStore
from ..core.types import WorldState

_inc = itertools.count(1)


def touching(a: np.ndarray, b: np.ndarray, margin: float) -> bool:
    """Do the boxes intersect after growing each by `margin` of its own size?
    (At impact, top-down boxes may overlap by only a few pixels.)"""
    def grow(x):
        w, h = x[2] - x[0], x[3] - x[1]
        return x + np.array([-w, -h, w, h], np.float32) * margin
    a, b = grow(np.asarray(a, np.float32)), grow(np.asarray(b, np.float32))
    return bool(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


class CollisionModule:
    def __init__(self, cfg: AppConfig, bus: EventBus, evidence: EvidenceStore, alerter) -> None:
        self.app = cfg
        self.cfg = cfg.collision
        self.bus = bus
        self.evidence = evidence
        self.alerter = alerter
        self._speeds: dict[tuple[str, int], deque] = {}
        self._cand: dict[tuple[str, int, int], float] = {}
        self._reported: set[tuple[str, int, int]] = set()
        self.incidents: list[dict] = []

    def _sudden_stop(self, key, t: float) -> bool:
        h = self._speeds.get(key)
        if not h:
            return False
        peak = max(v for ts, v in h if t - ts <= self.cfg.drop_window_s)
        return peak - h[-1][1] >= self.cfg.min_drop_kmph

    def update(self, world: WorldState, frames: dict) -> None:
        if not self.cfg.enabled:
            return
        t = world.t
        c = self.cfg
        alive = set()
        for a, st in world.approaches.items():
            tracks = [tr for tr in st.tracks if tr.confirmed]
            for tr in tracks:
                key = (a, tr.track_id)
                alive.add(key)
                h = self._speeds.setdefault(key, deque())
                if not h or h[-1][0] != t:
                    h.append((t, tr.speed_kmph))
                while h and t - h[0][0] > c.drop_window_s + c.confirm_s + 1:
                    h.popleft()
            if len(tracks) < 2:
                continue
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    p, q = tracks[i], tracks[j]
                    pk = (a, min(p.track_id, q.track_id), max(p.track_id, q.track_id))
                    if pk in self._reported:
                        continue
                    stopped = p.speed_kmph < c.stopped_kmph and q.speed_kmph < c.stopped_kmph
                    if not stopped or not touching(p.box, q.box, c.contact_margin):
                        self._cand.pop(pk, None)
                        continue
                    if pk not in self._cand:
                        if self._sudden_stop((a, p.track_id), t) or self._sudden_stop((a, q.track_id), t):
                            self._cand[pk] = t
                        continue
                    if t - self._cand[pk] >= c.confirm_s:
                        self._reported.add(pk)
                        self._cand.pop(pk, None)
                        self._collision(a, p, q, world, frames)
        if len(self._speeds) > 500:
            self._speeds = {k: v for k, v in self._speeds.items() if k in alive}

    # ------------------------------------------------------------------ incident
    def _collision(self, a: str, p, q, world: WorldState, frames: dict) -> None:
        t = world.t
        last = self.incidents[0] if self.incidents else None
        frame = frames.get(a)
        stem = f"collision_{a}_{p.track_id}_{q.track_id}_{time.strftime('%H%M%S')}"
        snap = None
        for tr in (p, q):
            tr.attrs["tag"] = "CRASH"
            tr.attrs["color_override"] = (30, 30, 255)
        if frame is not None:
            img = EvidenceStore.mark(frame, [p.box, q.box], "COLLISION",
                                     f"{self.app.junction.name} | {a} approach | {time.strftime('%Y-%m-%d %H:%M:%S')}",
                                     color=(30, 30, 255))
            snap = self.evidence.save_image(img, stem)
        vehicles = [f"{p.cls} #{p.track_id}", f"{q.cls} #{q.track_id}"]
        if last and t - last["t"] < self.cfg.merge_s:
            last["views"].append({"approach": a, "vehicles": vehicles, "snapshot": snap})
            self.bus.publish(Event(EventType.COLLISION,
                                   f"Same crash also seen by {a} camera ({' + '.join(vehicles)})",
                                   approach=a, severity=Severity.WARN,
                                   data={"incident": last["id"], "merged": True},
                                   evidence=[snap] if snap else []))
            return
        inc = {"id": f"INC-{time.strftime('%y%m%d-%H%M%S')}-{next(_inc)}", "t": round(t, 1), "ts": time.time(),
               "approach": a, "vehicles": vehicles, "snapshot": snap, "alert": None,
               "views": [{"approach": a, "vehicles": vehicles, "snapshot": snap}]}
        self.incidents = ([inc] + self.incidents)[:20]
        self.bus.publish(Event(EventType.COLLISION,
                               f"Collision on {a} approach: {' and '.join(vehicles)}. Calling ambulance",
                               approach=a, severity=Severity.CRITICAL,
                               data={"incident": inc["id"], "vehicles": vehicles},
                               evidence=[snap] if snap else []))
        self.evidence.submit(lambda: self._alert(inc))

    def _alert(self, inc: dict) -> None:
        loc = self.app.junction.location
        text = (f"ACCIDENT DETECTED - ambulance needed\n"
                f"Junction: {self.app.junction.name}\n"
                f"Approach: {inc['approach']}\n"
                f"Time: {time.strftime('%d %b %Y %H:%M:%S', time.localtime(inc['ts']))}\n"
                f"Vehicles: {', '.join(inc['vehicles'])}\n"
                f"Map: https://maps.google.com/?q={loc.lat},{loc.lon}\n"
                f"Incident: {inc['id']} (Traffic Agent)")
        photo = self.evidence.abs(inc["snapshot"]) if inc["snapshot"] else None
        res = self.alerter.send(text, photo, loc.lat, loc.lon, stem=f"alert_{inc['id']}")
        inc["alert"] = res
        if res["delivered"]:
            msg = f"Ambulance alert sent via {res['channel']} for {inc['id']}"
            sev = Severity.CRITICAL
        elif res["channel"] == "dry-run":
            msg = f"Ambulance alert for {inc['id']} recorded (dry run: Telegram not configured)"
            sev = Severity.WARN
        else:
            msg = f"Ambulance alert for {inc['id']} FAILED: {res.get('error')}"
            sev = Severity.CRITICAL
        ev = [res["record"]] if res.get("record") else []
        self.bus.publish(Event(EventType.AMBULANCE_CALLED, msg, approach=inc["approach"], severity=sev,
                               data={"incident": inc["id"], **res, "text": text}, evidence=ev))

    def status(self) -> dict:
        return {"enabled": self.cfg.enabled, "alerts": self.alerter.describe(),
                "count": len(self.incidents),
                "recent": [{k: i[k] for k in ("id", "t", "approach", "vehicles", "snapshot")}
                           | {"alert": (i["alert"] or {}).get("channel"),
                              "delivered": (i["alert"] or {}).get("delivered"),
                              "views": len(i["views"])} for i in self.incidents[:5]]}
