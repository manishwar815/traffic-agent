"""Click-to-calibrate tool: mark lanes, stop line and scale on a video frame.

  python scripts/draw_zones.py --video data/videos/vehicles.mp4 --out config/zones/N.json

Steps (instructions are shown on the image):
  1. Inbound lane     click polygon corners, ENTER to finish
  2. Inbound direction click 2 points: from -> to (the way traffic legally moves)
  3. Stop line        click 2 points across the lane           (S = skip)
  4. Outbound lane    polygon, ENTER                            (S = skip)
  5. Outbound direction 2 points                                (skipped with 4)
  6. Scale            click 2 points a known distance apart, then type the
                      distance in metres in the terminal        (S = skip, uses 10 px/m)
Keys: ENTER finish step, U undo point, S skip step, Q quit without saving.

The frame is resized to --width (must match `resize_width` in the config, default 640)
so the saved pixel coordinates line up with what the pipeline sees.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

STEPS = [
    ("in_poly", "poly", "1/6 INBOUND lane: click corners, ENTER to finish"),
    ("in_dir", "line", "2/6 INBOUND direction: click FROM then TO"),
    ("stop", "line", "3/6 STOP LINE: click 2 points across the lane (S = skip)"),
    ("out_poly", "poly", "4/6 OUTBOUND lane: click corners, ENTER (S = skip)"),
    ("out_dir", "line", "5/6 OUTBOUND direction: FROM then TO"),
    ("scale", "line", "6/6 SCALE: click 2 points a known distance apart (S = skip)"),
]


def grab_frame(path: str, index: int, width: int) -> np.ndarray:
    src = int(path) if path.isdigit() else path
    cap = cv2.VideoCapture(src)
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read a frame from {path}")
    if width and frame.shape[1] != width:
        frame = cv2.resize(frame, (width, int(round(frame.shape[0] * width / frame.shape[1]))))
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file, or webcam index like 0")
    ap.add_argument("--out", required=True, help="output JSON, e.g. config/zones/N.json")
    ap.add_argument("--frame", type=int, default=30, help="frame number to draw on")
    ap.add_argument("--width", type=int, default=640, help="same as resize_width in the config")
    args = ap.parse_args()

    base = grab_frame(args.video, args.frame, args.width)
    result: dict[str, list] = {}
    pts: list[tuple[int, int]] = []
    step = 0

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and step < len(STEPS):
            pts.append((x, y))

    win = "draw zones"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while step < len(STEPS):
        key_name, kind, text = STEPS[step]
        img = base.copy()
        for k, v in result.items():          # already finished steps
            col = (0, 200, 255) if k.startswith("in") else (200, 200, 200)
            if k == "stop":
                col = (0, 0, 255)
            if k == "scale":
                col = (255, 0, 255)
            arr = np.array(v, np.int32)
            if k.endswith("poly"):
                cv2.polylines(img, [arr], True, col, 2)
            else:
                cv2.arrowedLine(img, tuple(arr[0]), tuple(arr[1]), col, 2, tipLength=0.15)
        for i, p in enumerate(pts):          # current step
            cv2.circle(img, p, 4, (0, 255, 0), -1)
            if i:
                cv2.line(img, pts[i - 1], p, (0, 255, 0), 2)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF

        if kind == "line" and len(pts) == 2:
            result[key_name] = list(pts)
            pts.clear()
            step += 1
        elif key in (13, 10) and kind == "poly" and len(pts) >= 3:     # ENTER
            result[key_name] = list(pts)
            pts.clear()
            step += 1
        elif key in (ord("u"), ord("U")) and pts:
            pts.pop()
        elif key in (ord("s"), ord("S")) and key_name in ("stop", "out_poly", "scale"):
            pts.clear()
            step += 2 if key_name == "out_poly" else 1
        elif key in (ord("q"), ord("Q"), 27):
            print("Quit without saving.")
            return 1
    cv2.destroyAllWindows()

    px_per_m = 10.0
    if "scale" in result:
        (x1, y1), (x2, y2) = result["scale"]
        meters = float(input("Real distance between the two scale points (metres): ").strip() or "0")
        if meters > 0:
            px_per_m = math.hypot(x2 - x1, y2 - y1) / meters

    zones = {
        "px_per_m": round(px_per_m, 3),
        "anchor": "bottom",
        "lanes": [{"name": "inbound", "inbound": True, "polygon": result["in_poly"], "direction": result["in_dir"]}],
    }
    if "out_poly" in result and "out_dir" in result:
        zones["lanes"].append({"name": "outbound", "inbound": False, "polygon": result["out_poly"],
                               "direction": result["out_dir"]})
    if "stop" in result:
        zones["stop_line"] = result["stop"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(zones, indent=2))
    print(f"Saved {out}  (px_per_m = {px_per_m:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
