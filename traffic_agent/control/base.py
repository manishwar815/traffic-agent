"""Signal controller base: a phase cycle with one approach green at a time.

Cycle for each approach:  GREEN (duration chosen by the subclass)
                       -> AMBER (fixed) -> ALL-RED (fixed, clears the box)
                       -> next approach.

Subclasses decide only two things:
  * choose_next(world) -> which approach gets green next
  * green_time(approach, world) -> how long its green lasts
That keeps the safety logic (amber, all-red, min green) in one place.
"""
from __future__ import annotations

from typing import Optional

from ..core.types import WorldState


class PhaseCycleController:
    name = "base"

    def __init__(self, phase_order: list[str], amber_s: float = 3.0, all_red_s: float = 1.0) -> None:
        self.order = list(phase_order)
        self.amber_s = amber_s
        self.all_red_s = all_red_s
        self.current: str = self.order[-1]   # so the first choose_next() picks order[0]
        self.stage = "ALL_RED"               # GREEN / AMBER / ALL_RED
        self.stage_end = 0.0
        self.green_len = 0.0
        self.t = 0.0
        self.override: Optional[str] = None  # set by EmergencyModule: this approach gets green ASAP

    # ---- to be overridden ----------------------------------------------------
    def choose_next(self, world: WorldState) -> str:
        i = self.order.index(self.current)
        return self.order[(i + 1) % len(self.order)]

    def green_time(self, approach: str, world: WorldState) -> float:
        raise NotImplementedError

    # ---- state machine ---------------------------------------------------------
    def update(self, world: WorldState) -> dict[str, str]:
        self.t = world.t
        ov = self.override if self.override in self.order else None
        if ov and self.stage == "GREEN":
            if self.current == ov:
                self.stage_end = max(self.stage_end, self.t + 1.0)   # hold green while needed
            else:
                self.stage_end = min(self.stage_end, self.t)         # cut short: amber + all-red still run
        if self.t >= self.stage_end:
            if self.stage == "GREEN":
                self.stage, self.stage_end = "AMBER", self.t + self.amber_s
            elif self.stage == "AMBER":
                self.stage, self.stage_end = "ALL_RED", self.t + self.all_red_s
            else:
                self.current = ov or self.choose_next(world)
                self.green_len = self.green_time(self.current, world)
                self.stage, self.stage_end = "GREEN", self.t + self.green_len
        return self.states()

    def states(self) -> dict[str, str]:
        out = {a: "RED" for a in self.order}
        if self.stage == "GREEN":
            out[self.current] = "GREEN"
        elif self.stage == "AMBER":
            out[self.current] = "AMBER"
        return out

    def status(self) -> dict:
        return {"controller": self.name, "current": self.current, "stage": self.stage,
                "remaining_s": round(max(0.0, self.stage_end - self.t), 1),
                "green_len_s": round(self.green_len, 1), "override": self.override}
