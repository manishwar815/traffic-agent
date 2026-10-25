"""Arduino signal: real LEDs over USB serial, on top of the simulated signal.

The dashboard and event log keep working (SimulatedSignal still runs); every
state is also sent to an Arduino running hardware/arduino_signal/arduino_signal.ino.

Protocol (115200 baud, ASCII, one line per update):
  "GRRR\n"   one letter per approach in N E S W order: R / A / G, '-' = no lamp
The line is re-sent every heartbeat_s even when nothing changed. If the Arduino
hears nothing for its watchdog time (2 s) it flashes amber on all approaches,
so a crashed PC never leaves a stale green on the street.

Safety on this side too: a state with two greens is never sent (all red instead).
A missing or unplugged board is not fatal: we log it and retry every retry_s.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from ...config import APPROACHES, ArduinoCfg
from ...core.events import Event, EventBus, EventType, Severity
from .simulated import SimulatedSignal

LETTER = {"RED": "R", "AMBER": "A", "GREEN": "G"}


def encode(states: dict[str, str]) -> bytes:
    """{'N': 'GREEN', 'E': 'RED', ...} -> b'GR--\\n' (fail-safe: two greens -> all red)."""
    if sum(1 for s in states.values() if s == "GREEN") > 1:
        states = {a: "RED" for a in states}
    return ("".join(LETTER.get(states.get(a, ""), "-") for a in APPROACHES) + "\n").encode()


def _open_serial(port: str, baud: int):
    import serial                   # pyserial; only needed when the Arduino is enabled
    return serial.serial_for_url(port, baudrate=baud, timeout=0, write_timeout=0.5)


class ArduinoSignal(SimulatedSignal):
    def __init__(self, cfg: ArduinoCfg, bus: EventBus, world=None,
                 opener: Callable = _open_serial, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(bus, world)
        self.cfg = cfg
        self._opener = opener
        self._clock = clock
        self._port = None
        self._sent: Optional[bytes] = None
        self._last_send = -1e9
        self._last_try = -1e9
        self._warned = False

    def apply(self, states: dict[str, str]) -> None:
        super().apply(states)
        line = encode(states)
        now = self._clock()
        if line == self._sent and now - self._last_send < self.cfg.heartbeat_s:
            return
        if self._port is None and not self._connect(now):
            return
        try:
            self._port.write(line)
            self._sent, self._last_send = line, now
        except Exception as e:      # unplugged mid-run
            self._drop(f"Arduino on {self.cfg.port} lost ({e}); retrying every {self.cfg.retry_s:g}s")

    def _connect(self, now: float) -> bool:
        if now - self._last_try < self.cfg.retry_s:
            return False
        self._last_try = now
        try:
            self._port = self._opener(self.cfg.port, self.cfg.baud)
        except Exception as e:
            if not self._warned:    # say it once, not every retry
                self._system(f"Arduino not connected on {self.cfg.port} ({e}); "
                             f"lights shown on the dashboard only, retrying every {self.cfg.retry_s:g}s",
                             Severity.WARN)
                self._warned = True
            return False
        self._warned = False
        self._sent = None
        self._system(f"Arduino connected on {self.cfg.port}")
        return True

    def _drop(self, msg: str) -> None:
        self.close()
        self._system(msg, Severity.WARN)
        self._warned = True

    def _system(self, msg: str, severity: str = Severity.INFO) -> None:
        self.bus.publish(Event(EventType.SYSTEM, msg, severity=severity))

    def close(self) -> None:
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
