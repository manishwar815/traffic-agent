"""Download the pretrained ONNX models and free sample videos.

  python scripts/download_models.py            # models + sample videos
  python scripts/download_models.py --models   # models only

Models (all free, run on CPU with ONNX Runtime):
  yolo26n.onnx           Ultralytics YOLO26 nano, COCO classes (AGPL-3.0)
  plate_detector.onnx    licence-plate detector, open-image-models (MIT)
  plate_ocr.onnx (+yaml) licence-plate OCR, fast-plate-ocr global model (MIT)
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "models/yolo26n.onnx":
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.onnx",
    "models/plate_detector.onnx":
        "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx",
    "models/plate_ocr.onnx":
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.onnx",
    "models/plate_ocr.yaml":
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global_plate_config.yaml",
}

VIDEOS = {
    # Roboflow supervision example clips (traffic seen from above)
    "data/videos/vehicles.mp4": "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "data/videos/vehicles-2.mp4": "https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4",
    # Intel IoT sample videos (CC-BY 4.0)
    "data/videos/car-detection.mp4":
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
    "data/videos/person-bicycle-car-detection.mp4":
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
}


def fetch(rel: str, url: str) -> bool:
    dest = ROOT / rel
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  ok (exists)  {rel}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "traffic-agent/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done, shown = 0, -10
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = int(done * 100 / total) if total else 0
                if pct >= shown + 10:
                    shown = pct
                    print(f"\r  {pct:3d}%  {rel}", end="", flush=True)
        tmp.replace(dest)
        print(f"\r  ok ({dest.stat().st_size / 1e6:.1f} MB)  {rel}")
        return True
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"\r  FAILED  {rel}\n          {url}\n          {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="models only, skip videos")
    args = ap.parse_args()
    print("Models:")
    ok = all([fetch(k, v) for k, v in MODELS.items()])
    if not args.models:
        print("Sample videos (optional):")
        for k, v in VIDEOS.items():
            fetch(k, v)
    if not ok:
        print("\nSome models failed. Check your internet connection and run again.")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
