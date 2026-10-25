"""Evidence files: snapshots, challan PDFs, alert records.

Everything lands under storage.evidence_dir/<YYYY-MM-DD>/ and events store the
path RELATIVE to evidence_dir, which is what the dashboard serves at /evidence/.

Slow work (PDFs, network alerts) runs on one background worker thread so the
pipeline never stalls; call flush() to wait for it (tests, shutdown).
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


class EvidenceStore:
    def __init__(self, root: str | Path, jpeg_quality: int = 90) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = jpeg_quality
        self._jobs: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paths
    def new_path(self, stem: str, ext: str) -> tuple[Path, str]:
        """(absolute path, path relative to root) for a new evidence file."""
        day = time.strftime("%Y-%m-%d")
        rel = f"{day}/{stem}.{ext}"
        full = self.root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        return full, rel

    def abs(self, rel: str) -> Path:
        return self.root / rel

    # ------------------------------------------------------------------ images
    def save_image(self, img: np.ndarray, stem: str) -> str:
        full, rel = self.new_path(stem, "jpg")
        cv2.imwrite(str(full), img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return rel

    @staticmethod
    def mark(frame: np.ndarray, boxes: list, label: str, caption: str = "",
             color=(40, 40, 230)) -> np.ndarray:
        """Copy of frame with the offending vehicle(s) boxed and a caption bar."""
        img = frame.copy()
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 2, cv2.LINE_AA)
        if boxes:
            x1, y1 = int(boxes[0][0]), int(boxes[0][1])
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            ty = y1 - 8 if y1 - th - 10 > 0 else int(boxes[0][3]) + th + 8
            cv2.rectangle(img, (x1 - 3, ty - th - 4), (x1 + tw + 3, ty + 4), color, -1)
            cv2.putText(img, label, (x1, ty), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if caption:
            h = 22
            cv2.rectangle(img, (0, img.shape[0] - h), (img.shape[1], img.shape[0]), (20, 20, 20), -1)
            cv2.putText(img, caption, (6, img.shape[0] - 7), FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def crop(frame: np.ndarray, box, pad: float = 0.25, min_side: int = 24) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [float(v) for v in box]
        w, h = x2 - x1, y2 - y1
        H, W = frame.shape[:2]
        x1, x2 = int(max(0, x1 - pad * w)), int(min(W, x2 + pad * w))
        y1, y2 = int(max(0, y1 - pad * h)), int(min(H, y2 + pad * h))
        if x2 - x1 < min_side or y2 - y1 < min_side:
            return None
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------ background jobs
    def submit(self, fn: Callable[[], None]) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="evidence", daemon=True)
                self._worker.start()
        self._jobs.put(fn)

    def _run(self) -> None:
        while True:
            fn = self._jobs.get()
            try:
                fn()
            except Exception as exc:          # a broken PDF must not kill the worker
                print(f"[evidence] job failed: {exc!r}")
            finally:
                self._jobs.task_done()

    def flush(self, timeout: float = 30.0) -> None:
        """Wait until all queued jobs are done."""
        end = time.monotonic() + timeout
        while self._jobs.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
