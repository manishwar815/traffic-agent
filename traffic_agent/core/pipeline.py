"""The main loop: one shared view of the junction, every tick:

  sources -> detect -> track -> analyse -> WorldState
          -> modules (emergency, violations, collision) -> controller -> signal actuator
          -> annotated frames + state for the dashboard

Runs in a background thread; the web server reads `latest_jpeg` and `snapshot()`.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

from ..adapters.alerts.telegram import TelegramAlerter
from ..adapters.signal.arduino_serial import ArduinoSignal
from ..adapters.signal.simulated import SimulatedSignal
from ..config import AppConfig, resolve
from ..control.emergency import EmergencyModule
from ..control.fixed import make_controller
from ..enforcement.violations import ViolationModule, make_plate_reader
from ..incidents.collision import CollisionModule
from ..perception.analyzer import ApproachAnalyzer
from ..perception.tracker import Tracker
from ..perception.zones import Zones
from .annotate import annotate
from .events import Event, EventBus, EventType, Severity
from .evidence import EvidenceStore
from .types import ApproachState, WorldState


class Pipeline:
    def __init__(self, cfg: AppConfig, bus: EventBus) -> None:
        self.cfg = cfg
        self.bus = bus
        self.approaches = cfg.approaches
        self.sim = None
        self.sources = {}
        self.zones: dict[str, Zones] = {}
        self.detector = None
        self.detector_name = "simulated (ground-truth boxes)"
        self.frame_wh: dict[str, tuple[int, int]] = {}

        if cfg.mode == "sim":
            from ..sim.source import SimJunctionSource
            self.sim = SimJunctionSource(cfg.sim)
            for a in self.approaches:
                self.zones[a] = self.sim.zones(a)
                self.frame_wh[a] = (cfg.sim.view_width, cfg.sim.view_height)
        else:
            from ..io.video_source import VideoSource
            from ..perception.detector import YoloOnnxDetector
            d = cfg.detector
            self.detector = YoloOnnxDetector(resolve(d.model), d.input_size, d.conf, d.classes, d.threads)
            self.detector_name = f"{self.detector.name} (ONNX Runtime, CPU)"
            usable = []
            for a in self.approaches:
                sc = cfg.sources[a]
                if sc.type == "file" and not resolve(sc.path).exists():
                    print(f"[pipeline] WARNING: {a}: video not found ({sc.path}), approach skipped")
                    continue
                self.sources[a] = VideoSource(a, sc)
                usable.append(a)
            if not usable:
                raise SystemExit("No video sources found. Run: python scripts/download_models.py")
            self.approaches = usable

        self.trackers = {a: Tracker(cfg.tracker) for a in self.approaches}   # IDs are unique per approach
        self.analyzers: dict[str, ApproachAnalyzer] = {}
        # Only cycle through approaches that actually have a feed.
        sig_cfg = cfg.signal.model_copy(deep=True)
        sig_cfg.phase_order = [a for a in sig_cfg.phase_order if a in self.approaches] or list(self.approaches)
        self.controller = make_controller(sig_cfg)
        world = self.sim.world if self.sim else None
        self.actuator = (ArduinoSignal(cfg.signal.arduino, bus, world) if cfg.signal.arduino.enabled
                         else SimulatedSignal(bus, world))

        # ONNX Runtime releases the GIL, so the cameras' YOLO runs can overlap.
        self._det_pool = ThreadPoolExecutor(len(self.approaches), "detect") if self.detector else None
        self.emergency = EmergencyModule(cfg.emergency, bus, self.controller)
        self.evidence = EvidenceStore(resolve(cfg.storage.evidence_dir))
        self.violations = ViolationModule(cfg, bus, self.zones, self.evidence, make_plate_reader(cfg))
        self.collision = CollisionModule(cfg, bus, self.evidence, TelegramAlerter(cfg.alerts.telegram, self.evidence))
        self.modules: list = [self.emergency, self.violations, self.collision]
        self._commands: queue.SimpleQueue = queue.SimpleQueue()   # run on the pipeline thread
        self._demo_waiting: list = []                              # demo injections waiting for a condition
        self.latest_jpeg: dict[str, bytes] = {}
        self.world: Optional[WorldState] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_idx: dict[str, int] = {a: -1 for a in self.approaches}
        self._tick = 0
        self._fps = 0.0
        self._t0 = time.monotonic()
        self.signal = {a: "RED" for a in self.approaches}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "Pipeline":
        for s in self.sources.values():
            s.start()
        self._thread = threading.Thread(target=self.run, name="pipeline", daemon=True)
        self._thread.start()
        self.bus.publish(Event(EventType.SYSTEM, f"Pipeline started in {self.cfg.mode} mode "
                                                 f"({len(self.approaches)} approaches, detector: {self.detector_name})"))
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        for s in self.sources.values():
            s.stop()
        if self._det_pool:
            self._det_pool.shutdown(wait=False)
        self.actuator.close()
        self.evidence.flush(timeout=10)

    def run(self) -> None:
        period = 1.0 / self.cfg.sim.fps if self.sim else 0.0
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # keep running; report on the dashboard
                import traceback
                traceback.print_exc()
                self.bus.publish(Event(EventType.SYSTEM, f"Pipeline error: {exc!r}", severity=Severity.WARN))
                time.sleep(0.5)
            if self.sim and self.cfg.sim.realtime:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
            elif not self.sim:
                time.sleep(0.001)

    # ------------------------------------------------------------------ one tick
    def now(self) -> float:
        return self.sim.t if self.sim else time.monotonic() - self._t0

    def _get_frame(self, a: str) -> tuple[Optional[np.ndarray], bool]:
        """Returns (frame, is_new)."""
        if self.sim:
            return self.sim.frame(a), True
        idx, frame = self.sources[a].read()
        is_new = idx != self._frame_idx[a]
        self._frame_idx[a] = idx
        return frame, is_new

    def submit(self, fn) -> None:
        """Run fn() on the pipeline thread at the start of the next tick (thread-safe)."""
        self._commands.put(fn)

    def tick(self) -> None:
        t_start = time.perf_counter()
        while not self._commands.empty():
            self._commands.get_nowait()()
        self._demo_waiting = [fn for fn in self._demo_waiting if not fn()]
        if self.sim:
            self.sim.step(1.0 / self.cfg.sim.fps)
        t = self.now()
        frames: dict[str, np.ndarray] = {}
        states: dict[str, ApproachState] = {}
        any_new = False
        new: dict[str, bool] = {}
        for a in self.approaches:
            frame, is_new = self._get_frame(a)
            if frame is not None:
                frames[a], new[a] = frame, is_new
        # Detection for all cameras at once. With every_n_frames > 1 the cameras are
        # staggered, so each tick only runs YOLO on a share of them.
        n = max(1, self.cfg.detector.every_n_frames)
        to_detect = [a for i, a in enumerate(self.approaches)
                     if new.get(a) and (self._tick + i) % n == 0]
        dets_by: dict[str, list] = {}
        if self.sim:
            dets_by = {a: self.sim.detections(a) for a in to_detect}
        elif to_detect:
            dets_by = dict(zip(to_detect, self._det_pool.map(lambda a: self.detector(frames[a]), to_detect)))
        for a in self.approaches:
            if a not in frames:
                continue
            frame, is_new = frames[a], new[a]
            if a not in self.analyzers:            # lazy: need frame size for video
                h, w = frame.shape[:2]
                self.frame_wh[a] = (w, h)
                if a not in self.zones:
                    self.zones[a] = self._load_zones(a, w, h)
                self.analyzers[a] = ApproachAnalyzer(a, self.zones[a], self.cfg.analysis, (w, h))
            if is_new:
                any_new = True
                tracks = self.trackers[a].update(dets_by.get(a))
            else:
                tracks = self.trackers[a].active()
            if is_new:
                states[a] = self.analyzers[a].update(tracks, t)
            elif self.world and a in self.world.approaches:
                states[a] = self.world.approaches[a]
            else:
                states[a] = ApproachState(a, t)
        if not frames or (not self.sim and not any_new):
            time.sleep(0.005)
            return

        world = WorldState(t=t, approaches=states, signal=dict(self.signal), mode=self.cfg.mode)
        # --- Phase 2+ modules plug in here (they read world, publish events) ---
        for m in self.modules:
            m.update(world, frames)
        # --- signal control ---
        self.signal = self.controller.update(world)
        self.actuator.apply(self.signal)
        world.signal = dict(self.signal)
        for a, st in states.items():
            st.signal = self.signal.get(a, "RED")
            st.fps = self._fps

        # --- dashboard images ---
        q = [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.server.jpeg_quality]
        jpegs = {}
        for a, frame in frames.items():
            img = annotate(frame, self.zones[a], states[a])
            jpegs[a] = cv2.imencode(".jpg", img, q)[1].tobytes()
        if self.sim:
            jpegs["overview"] = cv2.imencode(".jpg", self.sim.overview(), q)[1].tobytes()
        dt = time.perf_counter() - t_start
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-3)) if self._fps else 1.0 / max(dt, 1e-3)
        with self._lock:
            self.latest_jpeg.update(jpegs)
            self.world = world
        self._tick += 1

    # ------------------------------------------------------------------ sim demo events
    DEMO_EVENTS = {
        "red_light": ("car", {"ignore_red": True}),
        "wrong_way": ("car", {"wrong_way": True}),
        "no_helmet": ("motorcycle", {"helmet": False}),
        "speeding": ("car", {"speeding": True}),
        "crash": ("car", {"distracted": True}),
        "ambulance": ("ambulance", {"siren": True}),
    }

    def demo_approach(self, kind: str) -> str:
        """Pick the approach where the injected event will actually happen soon."""
        world = self.world
        n = {a: (world.approaches[a].count_in_roi if world and a in world.approaches else 0) for a in self.approaches}
        q = {a: (world.approaches[a].queue_len if world and a in world.approaches else 0) for a in self.approaches}
        red = [a for a in self.approaches if self.signal.get(a) == "RED"] or list(self.approaches)
        green = [a for a in self.approaches if self.signal.get(a) == "GREEN"]
        if kind == "red_light":            # an empty red lane: the jumper reaches the line on red
            return min(red, key=lambda a: n[a])
        if kind == "speeding":             # a free road (inject_demo waits for a green one)
            w = self.sim.world
            return max(self.approaches, key=lambda a: (a in green, not w._uncrossed(a), -n[a]))
        if kind == "crash":                # a stopped queue on camera, else an empty lane
            w = self.sim.world
            return max(self.approaches, key=lambda a: (w.crash_score(a), a in red, q[a]))
        return min(self.approaches, key=lambda a: n[a])

    def inject_demo(self, kind: str, approach: Optional[str] = None) -> str:
        """Sim only: send a misbehaving vehicle into traffic. Returns the approach used."""
        if self.sim is None:
            raise RuntimeError("demo events need sim mode")
        vkind, flags = self.DEMO_EVENTS[kind]
        a = approach or self.demo_approach(kind)
        if kind == "crash":
            self.submit(lambda: self.sim.world.inject_crash(a))
        elif kind == "speeding" and approach is None:
            # Wait (up to 40 s) for a clear lane with green time left, so the
            # speeder is measured at full speed instead of braking for a red.
            deadline = self.now() + 40.0
            self.submit(lambda: self._demo_waiting.append(lambda: self._try_speeder(deadline)))
            return "next free green"
        elif kind == "speeding":
            self.submit(lambda: self.sim.world.inject_speeder(a))
        else:
            self.submit(lambda: self.sim.world.inject(a, vkind, **flags))
        return a

    def _try_speeder(self, deadline: float) -> bool:
        w = self.sim.world
        st = self.controller.status()
        free = [a for a in self.approaches if self.signal.get(a) == "GREEN" and st["remaining_s"] >= 3
                and w.speeder_room(a)]
        if free or self.now() >= deadline:
            w.inject_speeder(free[0] if free else self.demo_approach("speeding"))
            return True
        return False

    def _load_zones(self, a: str, w: int, h: int) -> Zones:
        path = self.cfg.sources[a].zones
        if path and resolve(path).exists():
            return Zones.load(resolve(path))
        self.bus.publish(Event(EventType.SYSTEM, f"[{a}] no zones file, using full frame. "
                                                 f"Run scripts/draw_zones.py for lanes/stop line.",
                               approach=a, severity=Severity.WARN))
        return Zones.full_frame(w, h)

    # ------------------------------------------------------------------ for the API
    def snapshot(self) -> dict:
        with self._lock:
            world = self.world
        if world is None:
            return {"ready": False}
        out = {
            "ready": True,
            "mode": self.cfg.mode,
            "junction": self.cfg.junction.name,
            "t": round(world.t, 1),
            "processing_fps": round(self._fps, 1),
            "detector": self.detector_name,
            "signal": world.signal,
            "controller": self.controller.status(),
            "emergency": self.emergency.status(),
            "violations": self.violations.status(),
            "collision": self.collision.status(),
            "approaches": {
                a: {"count": s.count_in_roi, "queue": s.queue_len, "occupancy": round(s.occupancy, 3),
                    "flow_per_min": s.flow_per_min, "tracks": len(s.tracks), "signal": s.signal}
                for a, s in world.approaches.items()},
            "feeds": list(self.latest_jpeg.keys()),
        }
        if self.sim:
            out["sim"] = self.sim.world.stats()
        else:
            out["sources"] = {a: {"connected": s.connected, "error": s.error} for a, s in self.sources.items()}
        return out

    def jpeg(self, name: str) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg.get(name)
