"""Adaptive controller: the busiest approach gets green next, for as long as it needs.

  * choose_next: highest vehicle count wins; an approach that has waited
    longer than max_red_s with vehicles jumps the line (no starvation).
  * green_time: min_green + per_vehicle_s * vehicles, capped at max_green.
  * gap-out: once min_green has passed, an empty approach ends its green early.
"""
from __future__ import annotations

from ..config import SignalCfg
from ..core.types import WorldState
from .base import PhaseCycleController


class AdaptiveController(PhaseCycleController):
    name = "adaptive"

    def __init__(self, cfg: SignalCfg) -> None:
        a = cfg.adaptive
        super().__init__(cfg.phase_order, a.amber_s, a.all_red_s)
        self.min_green = a.min_green_s
        self.max_green = a.max_green_s
        self.per_vehicle = a.per_vehicle_s
        self.max_red = a.max_red_s
        self.green_start = 0.0
        self.last_green = {ap: 0.0 for ap in self.order}   # when each approach last had green

    @staticmethod
    def _count(world: WorldState, approach: str) -> int:
        st = world.approaches.get(approach)
        return st.count_in_roi if st else 0

    def choose_next(self, world: WorldState) -> str:
        counts = {a: self._count(world, a) for a in self.order}
        waiting = [a for a in self.order if counts[a] > 0]
        # Don't give green straight back to the lane that just had it if others are waiting.
        others = [a for a in waiting if a != self.current]
        candidates = others or waiting
        if not candidates:
            # Nobody waiting: rotate to the approach that has been red the longest.
            return min(self.order, key=lambda a: self.last_green[a])
        starving = [a for a in candidates if self.t - self.last_green[a] >= self.max_red]
        if starving:
            return min(starving, key=lambda a: self.last_green[a])
        # Busiest first; ties go to whoever has been red longer.
        return max(candidates, key=lambda a: (counts[a], -self.last_green[a]))

    def green_time(self, approach: str, world: WorldState) -> float:
        n = self._count(world, approach)
        return min(self.max_green, self.min_green + self.per_vehicle * n)

    def update(self, world: WorldState) -> dict[str, str]:
        self.t = world.t
        if self.stage == "GREEN":
            self.last_green[self.current] = self.t
            # Gap-out: lane is empty after the minimum green, so stop wasting time.
            if (self.override != self.current
                    and self.t - self.green_start >= self.min_green
                    and self._count(world, self.current) == 0):
                self.stage_end = self.t
        prev_stage = self.stage
        out = super().update(world)
        if self.stage == "GREEN" and prev_stage != "GREEN":
            self.green_start = self.t
            self.last_green[self.current] = self.t
        return out
