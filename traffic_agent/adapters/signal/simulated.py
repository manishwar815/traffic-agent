"""Simulated signal: pushes the lights into the simulator world (if any)
and publishes a SIGNAL_CHANGE event whenever a light changes."""
from __future__ import annotations

from typing import Optional

from ...core.events import Event, EventBus, EventType
from .base import SignalActuator


class SimulatedSignal(SignalActuator):
    def __init__(self, bus: EventBus, world=None) -> None:
        self.bus = bus
        self.world = world              # sim.world.JunctionWorld or None (video mode)
        self.last: Optional[dict[str, str]] = None

    def apply(self, states: dict[str, str]) -> None:
        if self.world is not None:
            self.world.signal.update(states)
        if states != self.last:
            green = [a for a, s in states.items() if s == "GREEN"]
            amber = [a for a, s in states.items() if s == "AMBER"]
            if green:
                msg = f"Signal: {green[0]} GREEN"
            elif amber:
                msg = f"Signal: {amber[0]} AMBER"
            else:
                msg = "Signal: ALL RED"
            # Only GREEN starts are logged to the database to keep the log readable.
            self.bus.publish(Event(EventType.SIGNAL_CHANGE, msg, approach=(green or amber or [None])[0],
                                   data={"states": dict(states), "log": bool(green)}))
            self.last = dict(states)
