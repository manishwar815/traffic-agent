"""Entry point.

  python -m traffic_agent.main                          # simulator + dashboard
  python -m traffic_agent.main --config config/video.yaml   # real video files
  python -m traffic_agent.main --headless --duration 120    # no web server, prints stats
"""
from __future__ import annotations

import argparse
import json
import signal as _signal
import sys
import time

from .config import load_config, resolve
from .core.events import EventBus
from .core.pipeline import Pipeline
from .storage.db import Database


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AI traffic agent")
    p.add_argument("--config", default=None, help="YAML config (default: config/default.yaml)")
    p.add_argument("--mode", choices=["sim", "video"], help="override the config's mode")
    p.add_argument("--headless", action="store_true", help="run without the web dashboard")
    p.add_argument("--duration", type=float, default=0, help="headless: seconds of SIM time to run")
    p.add_argument("--fast", action="store_true", help="sim: run faster than real time")
    p.add_argument("--port", type=int, help="override dashboard port")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode
    if args.port:
        cfg.server.port = args.port
    if args.fast:
        cfg.sim.realtime = False

    bus = EventBus()
    db = Database(resolve(cfg.storage.db_path))
    bus.subscribe(db.on_event)
    bus.subscribe(lambda ev: print(f"[{ev.type}] {ev.message}") if ev.type != "SIGNAL_CHANGE" else None)

    pipeline = Pipeline(cfg, bus).start()

    if args.headless:
        try:
            limit = args.duration or float("inf")
            last_print = 0.0
            while pipeline.now() < limit:
                time.sleep(0.2)
                snap = pipeline.snapshot()
                if snap.get("ready") and snap["t"] - last_print >= 10:
                    last_print = snap["t"]
                    brief = {a: (v["count"], v["queue"]) for a, v in snap["approaches"].items()}
                    print(f"t={snap['t']:7.1f}s  fps={snap['processing_fps']:5.1f}  "
                          f"green={snap['controller']['current']}  (in lane, queue)={brief}"
                          + (f"  sim={snap['sim']}" if "sim" in snap else ""))
        except KeyboardInterrupt:
            pass
        finally:
            pipeline.stop()
            print(json.dumps(pipeline.snapshot().get("sim", {}), indent=None))
            db.close()
        return 0

    import uvicorn
    from .api.server import create_app
    app = create_app(cfg, pipeline, bus, db)
    print(f"\n  Dashboard: http://{cfg.server.host}:{cfg.server.port}\n  Press Ctrl+C to stop.\n")
    try:
        uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="warning")
    finally:
        pipeline.stop()
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
