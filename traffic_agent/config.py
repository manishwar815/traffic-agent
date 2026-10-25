"""Configuration loading and validation.

Every threshold and setting lives in a YAML file (config/default.yaml).
Pydantic validates it at startup, so a typo fails loudly instead of silently
using a wrong value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

APPROACHES = ("N", "E", "S", "W")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Strict(BaseModel):
    model_config = {"extra": "forbid"}  # unknown keys in YAML -> error


class Location(Strict):
    lat: float
    lon: float


class JunctionCfg(Strict):
    name: str = "Demo Junction"
    location: Location = Location(lat=17.4486, lon=78.3908)


class SourceCfg(Strict):
    """One camera feed (used when mode == 'video')."""
    type: Literal["file", "webcam", "rtsp"] = "file"
    path: str = ""                 # file path, webcam index ("0") or rtsp:// url
    loop: bool = True              # restart file at the end
    start_frame: int = 0           # lets you reuse one clip for several approaches
    zones: str = ""                # path to zones JSON made by scripts/draw_zones.py
    resize_width: int = 640        # frames are resized to this width


class SimCfg(Strict):
    seed: int = 42
    fps: float = 10.0
    realtime: bool = True                       # False = run as fast as possible
    arrivals_per_min: dict[str, float] = Field(
        default_factory=lambda: {"N": 9, "E": 3, "S": 6, "W": 2})
    vehicle_mix: dict[str, float] = Field(
        default_factory=lambda: {"car": 0.5, "motorcycle": 0.3, "bus": 0.08, "truck": 0.12})
    emergency_per_hour: float = 0.0             # random ambulances (siren on); 0 = only on demand
    speed_limit_kmph: float = 50.0
    arm_length_m: float = 80.0
    view_width: int = 640
    view_height: int = 360
    px_per_m: float = 12.0


class DetectorCfg(Strict):
    model: str = "models/yolo26n.onnx"
    input_size: int = 640
    conf: float = 0.25
    # COCO ids: 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
    classes: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 7])
    every_n_frames: int = 1        # run detection on every Nth frame (CPU saver)
    threads: int = 4               # ONNX Runtime intra-op threads


class TrackerCfg(Strict):
    high_thresh: float = 0.5       # confident detections, matched first
    low_thresh: float = 0.1        # weak detections, used to keep existing tracks alive
    new_track_thresh: float = 0.5  # min score to start a new track
    match_iou: float = 0.2         # min IoU to accept a match
    max_lost: int = 15             # frames a track survives without a match
    min_hits: int = 2              # matches needed before a track is "confirmed"


class AnalysisCfg(Strict):
    queue_speed_kmph: float = 5.0  # slower than this inside ROI = queued
    speed_window_s: float = 1.0    # time window for speed estimate


class FixedSignalCfg(Strict):
    green_s: float = 20.0
    amber_s: float = 3.0
    all_red_s: float = 1.0


class AdaptiveSignalCfg(Strict):
    min_green_s: float = 8.0       # never shorter than this (unless nobody is waiting anywhere)
    max_green_s: float = 45.0      # never longer than this
    per_vehicle_s: float = 2.5     # extra green per vehicle in the lane
    max_red_s: float = 90.0        # a lane red this long with vehicles gets priority
    amber_s: float = 3.0
    all_red_s: float = 1.0


class ArduinoCfg(Strict):
    enabled: bool = False          # true = also drive real LEDs (hardware/arduino_signal/)
    port: str = "/dev/ttyACM0"     # Windows: "COM3" etc.
    baud: int = 115200
    heartbeat_s: float = 0.5       # re-send this often; the Arduino flashes amber after 2 s of silence
    retry_s: float = 5.0           # reconnect attempt interval when the board is missing


class SignalCfg(Strict):
    controller: Literal["fixed", "adaptive"] = "adaptive"
    phase_order: list[str] = Field(default_factory=lambda: list(APPROACHES))
    fixed: FixedSignalCfg = FixedSignalCfg()
    adaptive: AdaptiveSignalCfg = AdaptiveSignalCfg()
    arduino: ArduinoCfg = ArduinoCfg()


class EmergencyCfg(Strict):
    enabled: bool = True
    max_hold_s: float = 60.0       # a single request never holds priority longer than this
    lost_grace_s: float = 2.0      # vehicle unseen this long = it has passed
    manual_hold_s: float = 20.0    # green held for a manual (dashboard / API) request


class FineCfg(Strict):
    """One violation type: is it enforced, what it costs, which law it breaks."""
    enabled: bool = True
    amount: int = 1000             # rupees
    section: str = ""


class ViolationsCfg(Strict):
    enabled: bool = True
    red_light: FineCfg = FineCfg(amount=1000, section="MV Act s.119 / s.177A (jumping red light)")
    wrong_way: FineCfg = FineCfg(amount=1000, section="MV Act s.184 (dangerous driving: wrong side)")
    no_helmet: FineCfg = FineCfg(amount=1000, section="MV Act s.129 / s.194D (riding without helmet)")
    speeding: FineCfg = FineCfg(amount=2000, section="MV Act s.112 / s.183 (over-speeding)")
    red_grace_s: float = 1.5       # crossing within this long after red starts is not a violation
    wrong_way_min_s: float = 1.0   # must move against the lane direction this long
    min_speed_kmph: float = 8.0    # slower than this: no wrong-way / heading decision
    helmet_min_frames: int = 3     # "no helmet" seen in this many frames before fining
    speed_limit_kmph: float = 50.0
    speed_tolerance_pct: float = 10.0
    speed_min_s: float = 0.5       # over the limit (with tolerance) for this long
    dedup_s: float = 300.0         # same plate + same violation within this window = one challan
    read_plates: bool = True       # video mode: run the ANPR models on violators
    plate_detector: str = "models/plate_detector.onnx"
    plate_ocr: str = "models/plate_ocr.onnx"


class CollisionCfg(Strict):
    enabled: bool = True
    contact_margin: float = 0.05   # boxes grown by this fraction of their size must touch
    min_drop_kmph: float = 20.0    # one of them lost at least this much speed ...
    drop_window_s: float = 1.5     # ... within this long (sudden stop, not braking)
    stopped_kmph: float = 4.0      # "stationary" means slower than this
    confirm_s: float = 2.0         # both stay stationary and overlapping this long -> collision
    merge_s: float = 60.0          # another camera seeing the same crash within this window = same incident


class TelegramCfg(Strict):
    enabled: bool = True
    bot_token: str = ""            # or env TELEGRAM_BOT_TOKEN
    chat_id: str = ""              # or env TELEGRAM_CHAT_ID
    timeout_s: float = 10.0


class AlertsCfg(Strict):
    telegram: TelegramCfg = TelegramCfg()


class StorageCfg(Strict):
    db_path: str = "data/traffic.db"
    evidence_dir: str = "evidence"


class ServerCfg(Strict):
    host: str = "127.0.0.1"
    port: int = 8000
    jpeg_quality: int = 70


class AppConfig(Strict):
    mode: Literal["sim", "video"] = "sim"
    junction: JunctionCfg = JunctionCfg()
    sources: dict[str, SourceCfg] = Field(default_factory=dict)
    sim: SimCfg = SimCfg()
    detector: DetectorCfg = DetectorCfg()
    tracker: TrackerCfg = TrackerCfg()
    analysis: AnalysisCfg = AnalysisCfg()
    signal: SignalCfg = SignalCfg()
    emergency: EmergencyCfg = EmergencyCfg()
    violations: ViolationsCfg = ViolationsCfg()
    collision: CollisionCfg = CollisionCfg()
    alerts: AlertsCfg = AlertsCfg()
    storage: StorageCfg = StorageCfg()
    server: ServerCfg = ServerCfg()

    @model_validator(mode="after")
    def _check(self) -> "AppConfig":
        for a in self.sources:
            if a not in APPROACHES:
                raise ValueError(f"sources: unknown approach '{a}', use one of {APPROACHES}")
        if self.mode == "video" and not self.sources:
            raise ValueError("mode is 'video' but no sources are configured")
        for a in self.signal.phase_order:
            if a not in APPROACHES:
                raise ValueError(f"signal.phase_order: unknown approach '{a}'")
        return self

    @property
    def approaches(self) -> list[str]:
        """Approaches that actually have a feed."""
        if self.mode == "sim":
            return list(APPROACHES)
        return [a for a in APPROACHES if a in self.sources]


def resolve(path: str) -> Path:
    """Relative paths in the config are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path) if path else PROJECT_ROOT / "config" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)
