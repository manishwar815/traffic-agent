"""Threaded camera / video-file reader.

Why a thread: cv2.VideoCapture.read() blocks. For live cameras we want the
*latest* frame, not a growing backlog, so a background thread keeps reading
and we only keep the newest frame. For files we read in order (no skipping)
but pace playback to the file's FPS so the demo looks real-time.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..config import SourceCfg, resolve


class VideoSource:
    def __init__(self, name: str, cfg: SourceCfg) -> None:
        self.name = name
        self.cfg = cfg
        self.is_file = cfg.type == "file"
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_idx = -1           # increases each time a new frame arrives
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.fps = 25.0
        self.connected = False
        self.error: Optional[str] = None

    # ---------------------------------------------------------------- open
    def _open(self) -> cv2.VideoCapture:
        if self.cfg.type == "webcam":
            cap = cv2.VideoCapture(int(self.cfg.path or 0))
        elif self.cfg.type == "rtsp":
            cap = cv2.VideoCapture(self.cfg.path, cv2.CAP_FFMPEG)
        else:
            path = resolve(self.cfg.path)
            if not path.exists():
                raise FileNotFoundError(f"[{self.name}] video file not found: {path}")
            cap = cv2.VideoCapture(str(path))
            if self.cfg.start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.cfg.start_frame)
        if not cap.isOpened():
            raise RuntimeError(f"[{self.name}] could not open source: {self.cfg.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 120 else 25.0
        return cap

    def start(self) -> "VideoSource":
        self._cap = self._open()
        self.connected = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"src-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    # ---------------------------------------------------------------- loop
    def _resize(self, frame: np.ndarray) -> np.ndarray:
        w = self.cfg.resize_width
        if w and frame.shape[1] != w:
            h = int(round(frame.shape[0] * w / frame.shape[1]))
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame

    def _loop(self) -> None:
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if self.is_file and self.cfg.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.cfg.start_frame)
                    continue
                # Live stream dropped: try to reconnect every 2 s.
                self.connected = False
                self.error = "stream ended / disconnected"
                time.sleep(2)
                try:
                    self._cap.release()
                    self._cap = self._open()
                    self.connected = True
                    self.error = None
                except Exception as exc:
                    self.error = str(exc)
                continue
            frame = self._resize(frame)
            with self._lock:
                self._frame = frame
                self._frame_idx += 1
            if self.is_file:   # pace file playback to real time
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()

    # ---------------------------------------------------------------- read
    def read(self) -> tuple[int, Optional[np.ndarray]]:
        """Return (frame_index, frame). frame_index lets callers skip repeats."""
        with self._lock:
            return self._frame_idx, self._frame
