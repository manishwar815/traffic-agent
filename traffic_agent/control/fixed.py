"""Fixed-time controller: every approach gets the same green, in a fixed order.
This is how most Indian junctions run today, and it is our baseline for
measuring the adaptive controller in Phase 2."""
from __future__ import annotations

from ..config import SignalCfg
from ..core.types import WorldState
from .base import PhaseCycleController


class FixedTimeController(PhaseCycleController):
    name = "fixed"

    def __init__(self, cfg: SignalCfg) -> None:
        super().__init__(cfg.phase_order, cfg.fixed.amber_s, cfg.fixed.all_red_s)
        self.green_s = cfg.fixed.green_s

    def green_time(self, approach: str, world: WorldState) -> float:
        return self.green_s


def make_controller(cfg: SignalCfg) -> PhaseCycleController:
    if cfg.controller == "fixed":
        return FixedTimeController(cfg)
    if cfg.controller == "adaptive":
        from .adaptive import AdaptiveController
        return AdaptiveController(cfg)
    raise ValueError(f"controller '{cfg.controller}' arrives in Phase 2 (traffic management)")
