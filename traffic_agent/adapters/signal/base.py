"""Signal actuator interface: the swap point between simulation and real hardware.

The controller DECIDES the lights; an actuator APPLIES them somewhere:
  * SimulatedSignal  -> the built-in simulator + dashboard (now)
  * ArduinoSignal    -> real LEDs over USB serial (signal.arduino in the YAML)
  * a real junction  -> a traffic-controller protocol (e.g. NTCIP), not in scope
All of them implement apply(); nothing else in the system changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class SignalActuator(ABC):
    @abstractmethod
    def apply(self, states: dict[str, str]) -> None:
        """states: approach -> 'RED' | 'AMBER' | 'GREEN'"""

    def close(self) -> None:
        pass
