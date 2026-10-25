"""Events and a tiny thread-safe publish/subscribe bus.

Modules publish Events ("emergency vehicle on N", "red light jump by #12").
Subscribers (database writer, dashboard stream, action adapters) receive them.
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


class EventType:
    SYSTEM = "SYSTEM"
    SIGNAL_CHANGE = "SIGNAL_CHANGE"
    SIGNAL_PLAN = "SIGNAL_PLAN"          # Phase 2: adaptive timing decision
    EMERGENCY = "EMERGENCY"              # Phase 2b
    VIOLATION = "VIOLATION"              # Phase 2c
    FINE_ISSUED = "FINE_ISSUED"          # Phase 2c
    COLLISION = "COLLISION"              # Phase 2d
    AMBULANCE_CALLED = "AMBULANCE_CALLED"  # Phase 2d


class Severity:
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


_ids = itertools.count(1)


@dataclass
class Event:
    type: str
    message: str
    approach: Optional[str] = None
    severity: str = Severity.INFO
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)   # file paths (images, clips, PDFs)
    ts: float = field(default_factory=time.time)        # wall-clock time
    id: int = field(default_factory=lambda: next(_ids))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    """Synchronous callbacks + per-client queues (for the dashboard stream)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[Event], None]] = []
        self._queues: list[queue.Queue] = []

    def subscribe(self, cb: Callable[[Event], None]) -> None:
        with self._lock:
            self._callbacks.append(cb)

    def open_queue(self, maxsize: int = 500) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._queues.append(q)
        return q

    def close_queue(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    def publish(self, event: Event) -> Event:
        with self._lock:
            callbacks = list(self._callbacks)
            queues = list(self._queues)
        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:  # one bad subscriber must not kill the pipeline
                print(f"[events] subscriber error: {exc!r}")
        for q in queues:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow dashboard client: drop rather than block
        return event
