"""Emergency vehicle priority (green wave for ambulances, fire trucks, police).

Where requests come from:
  * a confirmed track with attrs["emergency"] = True. In sim mode the simulator
    sets it for an ambulance with its siren on. In video mode a future classifier
    (custom YOLO class, siren audio...) sets it: plain COCO YOLO cannot tell an
    ambulance from a van.
  * a manual request (dashboard button / API), standing in for GPS, RFID or an
    operator in the control room.

What happens: the earliest open request wins and becomes controller.override.
The controller then cuts the current green short (amber + all-red still run, so
it stays safe), gives green to that approach, and holds it until the vehicle has
crossed the stop line or left the camera. A request never holds green longer
than max_hold_s, so a false detection cannot block the junction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import EmergencyCfg
from ..core.events import Event, EventBus, EventType, Severity
from ..core.types import WorldState
from .base import PhaseCycleController


@dataclass
class EmergencyRequest:
    approach: str
    source: str            # "track #12" or "manual"
    since: float
    last_seen: float
    expires: Optional[float] = None   # manual requests only


class EmergencyModule:
    def __init__(self, cfg: EmergencyCfg, bus: EventBus, controller: PhaseCycleController) -> None:
        self.cfg = cfg
        self.bus = bus
        self.controller = controller
        self.requests: dict[tuple[str, object], EmergencyRequest] = {}
        self._finished: set[tuple[str, object]] = set()   # track keys already served / timed out
        self._manual: list[tuple[str, float]] = []        # queued from the API thread
        self.served: list[dict] = []                      # history, for stats

    def request_manual(self, approach: str, hold_s: Optional[float] = None) -> None:
        """Called from the API (any thread); picked up on the next update()."""
        self._manual.append((approach, hold_s or self.cfg.manual_hold_s))

    @property
    def active(self) -> Optional[EmergencyRequest]:
        return min(self.requests.values(), key=lambda r: r.since) if self.requests else None

    def update(self, world: WorldState, frames: dict) -> None:
        if not self.cfg.enabled:
            return
        t = world.t
        while self._manual:
            a, hold = self._manual.pop(0)
            if a in world.approaches:
                self._open((a, "manual"), EmergencyRequest(a, "manual", t, t, expires=t + hold))

        for a, st in world.approaches.items():
            for tr in st.tracks:
                # Only inbound and not yet past the stop line: a camera also sees the
                # ambulance driving AWAY after it crossed the junction from another arm.
                if not (tr.attrs.get("emergency") and tr.confirmed and tr.in_roi) or tr.crossed_at is not None:
                    continue
                key = (a, tr.track_id)
                if key in self._finished:
                    continue
                if key in self.requests:
                    self.requests[key].last_seen = t
                else:
                    self._open(key, EmergencyRequest(a, f"track #{tr.track_id}", t, t))

        for key, r in list(self.requests.items()):
            if r.expires is not None:
                if t >= r.expires:
                    self._close(key, t, "manual hold ended")
            elif t - r.last_seen > self.cfg.lost_grace_s:
                self._close(key, t, "vehicle has passed")       # crossed the stop line or left view
            elif t - r.since > self.cfg.max_hold_s:
                self._close(key, t, f"timed out after {self.cfg.max_hold_s:.0f} s", Severity.WARN)

        act = self.active
        self.controller.override = act.approach if act else None

    def _open(self, key, r: EmergencyRequest) -> None:
        self.requests[key] = r
        self.bus.publish(Event(EventType.EMERGENCY,
                               f"Emergency vehicle on {r.approach} ({r.source}): giving priority green",
                               approach=r.approach, severity=Severity.CRITICAL,
                               data={"state": "start", "source": r.source}))

    def _close(self, key, t: float, why: str, severity: str = Severity.INFO) -> None:
        r = self.requests.pop(key)
        if r.expires is None:
            self._finished.add(key)
        self.served.append({"approach": r.approach, "source": r.source, "since": r.since,
                            "duration_s": round(t - r.since, 1), "why": why})
        self.bus.publish(Event(EventType.EMERGENCY,
                               f"Emergency on {r.approach} cleared after {t - r.since:.0f} s ({why})",
                               approach=r.approach, severity=severity,
                               data={"state": "end", "source": r.source, "duration_s": round(t - r.since, 1)}))

    def status(self) -> dict:
        act = self.active
        return {"enabled": self.cfg.enabled,
                "active": None if act is None else {"approach": act.approach, "source": act.source,
                                                    "since": round(act.since, 1)},
                "pending": len(self.requests), "served": len(self.served)}
