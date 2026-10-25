# Traffic Agent: AI Smart Traffic Monitoring & Enforcement

An "intelligent traffic agent" that watches a 4-way junction through cameras
(or a built-in simulator), measures traffic, and acts on its own.

| Module | Status |
|---|---|
| Foundation: input, YOLO detection, tracking, zones, dashboard | **Phase 1, done** |
| 1. Adaptive signal timing | **Phase 2, done** |
| 2. Emergency vehicle green priority | **Phase 2, done** |
| 3. Violations + e-challan (red light, wrong way, no helmet, speeding) | **Phase 2, done** |
| 4. Collision detection + ambulance call (Telegram) | **Phase 2, done** |

## Two run modes, one pipeline

| | `sim` (default) | `video` |
|---|---|---|
| Input | Built-in junction simulator, 4 camera views | Video files / webcam / RTSP, one per approach |
| Detector | Simulated boxes from the simulator | **YOLO26n** on ONNX Runtime (CPU) |
| Signal | Vehicles obey our signal (closed loop) | Virtual signal (recorded cars can't obey it) |
| Use it for | Live demo, measuring waiting time, injecting events | Showing real computer vision on real footage |

Everything after detection (tracking, measurement, all modules, dashboard) is the same code in both modes.

## Setup: Linux Mint / Ubuntu

```bash
sudo apt install -y python3-venv python3-pip      # once
cd traffic-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py                 # ~21 MB of models + sample videos
```

## Setup: Windows (PowerShell, Python 3.10 to 3.12 from python.org)

```powershell
cd traffic-agent
py -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python scripts\download_models.py
```

## Run

```bash
python -m traffic_agent.main                          # simulator + dashboard
python -m traffic_agent.main --config config/video.yaml   # real video with YOLO26n
```

Open **http://127.0.0.1:8000** in your browser.

### Demo the Phase 2 modules (sim mode)

The dashboard has buttons for each event. The same actions are available over the API:

```bash
curl -X POST http://127.0.0.1:8000/api/sim/ambulance/N        # ambulance with siren -> priority green
curl -X POST http://127.0.0.1:8000/api/emergency/E            # manual emergency hold (GPS / RFID stand-in)
curl -X POST http://127.0.0.1:8000/api/sim/inject/red_light   # also: wrong_way | no_helmet | speeding | crash
curl http://127.0.0.1:8000/api/fines                          # issued e-challans (?plate=...)
curl http://127.0.0.1:8000/api/fines/summary
```

Each violation saves an evidence snapshot and a PDF e-challan under `evidence/<day>/`. If the plate
can't be read, the challan goes to manual review (`POST /api/fines/{no}` sets the plate or status).

### Ambulance alerts on Telegram

1. In Telegram, talk to @BotFather, send `/newbot`, and copy the bot token.
2. Send any message to the bot, or add it to the control-room group.
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `"chat":{"id": ...}`.
4. `export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...` (or set them in the YAML).

Without a token the alerter runs in dry-run mode. It writes each alert to `evidence/<day>/alert_*.json`,
so the demo works offline.

### Real traffic lights on an Arduino (optional)

1. Wire 12 LEDs (each through a 220 Ω resistor to GND) to an Uno or Nano: N on pins 2/3/4,
   E on 5/6/7, S on 8/9/10, W on 11/12/13 (red/amber/green).
2. Flash `hardware/arduino_signal/arduino_signal.ino` with the Arduino IDE.
3. In the YAML, set `signal.arduino: {enabled: true, port: /dev/ttyACM0}` (Windows: `COM3`).
   On Linux, add yourself to the serial group once: `sudo usermod -aG dialout $USER`, then log out and back in.

The lights follow the dashboard. The board flashes amber when it hears nothing from the PC for 2 s,
or when a command asks for two greens. If the board is unplugged, the system keeps running and reconnects.

Other options:

```bash
python -m traffic_agent.main --headless --fast --duration 600   # 10 sim-minutes, prints stats, no browser
python -m pytest -q                                             # unit tests
```

## Calibrate a real camera (video mode)

```bash
python scripts/draw_zones.py --video data/videos/vehicles.mp4 --out config/zones/N.json
```

Click the inbound lane, its direction, the stop line, the optional outbound lane, and two points a known
distance apart (for speed). Without a zones file, the whole frame counts as one lane: speeds are not meaningful
and the wrong-way rule is switched off.

## Project layout

```
config/            default.yaml (sim), video.yaml, zones/*.json
traffic_agent/
  main.py          entry point
  config.py        YAML + validation
  core/            types, events (pub/sub), pipeline loop, annotation
  io/              threaded video/webcam/RTSP reader
  sim/             junction simulator (IDM car-following), renderer, sim camera source
  perception/      YOLO ONNX detector, ByteTrack-style tracker, zones, per-approach analyzer
  control/         signal controllers: fixed, adaptive, emergency priority
  enforcement/     violation rules, PDF e-challans
  incidents/       collision detection
  adapters/alerts/ Telegram alerter (dry-run without a token)
  adapters/signal/ actuator interface: simulated, Arduino over USB serial
  storage/         SQLite event log + fines
  api/             Starlette server + static dashboard (HTML/JS/CSS)
hardware/          Arduino sketch for the LED junction
scripts/           download_models.py, draw_zones.py
tests/             pytest unit tests
```

## Models and licences

| File | What | Licence |
|---|---|---|
| `models/yolo26n.onnx` | Ultralytics YOLO26 nano, COCO vehicles | AGPL-3.0 (fine for academic use) |
| `models/plate_detector.onnx` | Licence-plate detector (open-image-models) | MIT |
| `models/plate_ocr.onnx` | Licence-plate OCR (fast-plate-ocr) | MIT |
