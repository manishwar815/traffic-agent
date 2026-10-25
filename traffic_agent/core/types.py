"""Shared data types passed between layers.

Flow per tick:  frame -> Detection list -> Track list -> ApproachState -> WorldState
Every module only reads WorldState, so modules never care whether the data
came from a real camera (YOLO) or from the built-in simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Class names we use everywhere (COCO ids map onto these).
COCO_TO_NAME = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle", "ambulance")


@dataclass
class Detection:
    """One object found in one frame."""
    box: np.ndarray            # [x1, y1, x2, y2] in frame pixels (float)
    score: float
    cls: str                   # e.g. "car"
    # Extra info. From the simulator this holds ground truth (plate, helmet...).
    # From YOLO it starts empty and later layers (plate reader...) fill it in.
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """One vehicle followed over time on one approach camera."""
    track_id: int
    box: np.ndarray
    cls: str
    score: float
    attrs: dict[str, Any] = field(default_factory=dict)
    hits: int = 1               # number of frames matched
    lost: int = 0               # frames since last match
    confirmed: bool = False
    # Filled by the analyzer:
    history: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x_m, y_m) ground point
    speed_kmph: float = 0.0
    heading_ok: Optional[bool] = None   # None = not enough motion to tell
    in_roi: bool = False
    crossed_stop_line: bool = False     # crossed during THIS tick
    crossed_at: Optional[float] = None  # time it crossed (sticky)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the vehicle touches the road."""
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, y2


@dataclass
class ApproachState:
    """Everything known about one approach (N/E/S/W) at one instant."""
    name: str
    t: float
    tracks: list[Track] = field(default_factory=list)
    count_in_roi: int = 0          # vehicles currently in the lane area
    queue_len: int = 0             # of those, how many are (nearly) stopped
    occupancy: float = 0.0         # fraction of ROI area covered by boxes (0..1)
    flow_per_min: float = 0.0      # stop-line crossings in the last minute
    signal: str = "RED"            # signal currently shown to this approach
    fps: float = 0.0


@dataclass
class WorldState:
    t: float
    approaches: dict[str, ApproachState]
    signal: dict[str, str]         # approach -> RED / AMBER / GREEN
    mode: str = "sim"
