"""Web dashboard backend (Starlette + Uvicorn).

Endpoints
  GET  /                    dashboard page
  GET  /frame/{name}.jpg    latest annotated frame (N, E, S, W, overview)
  GET  /api/state           live state snapshot (JSON)
  GET  /api/events          recent events from the database
  GET  /events/stream       Server-Sent Events: live events + state twice a second
  GET  /evidence/...        evidence files (snapshots, clips, challan PDFs)
  POST /api/emergency/{a}   manual emergency priority for approach a (GPS / RFID / operator stand-in)
  POST /api/sim/ambulance/{a}  sim only: send an ambulance with siren into approach a
  POST /api/sim/inject/{kind}  sim only: red_light | wrong_way | no_helmet | speeding | crash
                            (?approach=N, else the approach where it will happen soonest)
  GET  /api/fines           e-challans (?plate=TS09EC6993&limit=50)
  GET  /api/fines/summary   totals per violation
  POST /api/fines/{no}      {"status": "PAID" | "CANCELLED" | "ISSUED", "plate": "..."} (review desk)

Why polling JPEGs instead of MJPEG: browsers allow only ~6 open connections per
host; 5 never-ending video streams + the event stream would hit that limit.
"""
from __future__ import annotations

import asyncio
import json
import queue
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..config import AppConfig, resolve
from ..core.events import EventBus
from ..core.pipeline import Pipeline
from ..storage.db import Database

STATIC = Path(__file__).parent / "static"
NO_CACHE = {"Cache-Control": "no-store"}


def create_app(cfg: AppConfig, pipeline: Pipeline, bus: EventBus, db: Database) -> Starlette:
    evidence_dir = resolve(cfg.storage.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    async def index(request: Request):
        return FileResponse(STATIC / "index.html", headers=NO_CACHE)

    async def frame(request: Request):
        data = pipeline.jpeg(request.path_params["name"])
        if data is None:
            return Response(status_code=404)
        return Response(data, media_type="image/jpeg", headers=NO_CACHE)

    async def state(request: Request):
        return JSONResponse(pipeline.snapshot(), headers=NO_CACHE)

    async def events(request: Request):
        limit = int(request.query_params.get("limit", 100))
        return JSONResponse(db.recent_events(limit=limit), headers=NO_CACHE)

    async def stream(request: Request):
        q = bus.open_queue()

        async def gen():
            try:
                tick = 0
                while not await request.is_disconnected():
                    while True:
                        try:
                            ev = q.get_nowait()
                        except queue.Empty:
                            break
                        yield f"event: ev\ndata: {json.dumps(ev.to_dict(), default=str)}\n\n"
                    if tick % 5 == 0:
                        yield f"event: state\ndata: {json.dumps(pipeline.snapshot(), default=str)}\n\n"
                    tick += 1
                    await asyncio.sleep(0.1)
            finally:
                bus.close_queue(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def emergency(request: Request):
        a = request.path_params["approach"]
        if a not in pipeline.approaches:
            return JSONResponse({"error": f"unknown approach {a}"}, status_code=404)
        pipeline.emergency.request_manual(a)
        return JSONResponse({"ok": True, "approach": a})

    async def sim_ambulance(request: Request):
        a = request.path_params["approach"]
        if pipeline.sim is None:
            return JSONResponse({"error": "only available in sim mode"}, status_code=400)
        if a not in pipeline.approaches:
            return JSONResponse({"error": f"unknown approach {a}"}, status_code=404)
        pipeline.submit(lambda: pipeline.sim.world.inject(a, "ambulance", siren=True))
        return JSONResponse({"ok": True, "approach": a})

    async def sim_inject(request: Request):
        kind = request.path_params["kind"]
        if pipeline.sim is None:
            return JSONResponse({"error": "only available in sim mode"}, status_code=400)
        if kind not in pipeline.DEMO_EVENTS:
            return JSONResponse({"error": f"unknown event {kind}"}, status_code=404)
        a = request.query_params.get("approach")
        if a and a not in pipeline.approaches:
            return JSONResponse({"error": f"unknown approach {a}"}, status_code=404)
        return JSONResponse({"ok": True, "kind": kind, "approach": pipeline.inject_demo(kind, a)})

    async def fines(request: Request):
        limit = int(request.query_params.get("limit", 50))
        return JSONResponse(db.recent_fines(limit, request.query_params.get("plate")), headers=NO_CACHE)

    async def fines_summary(request: Request):
        return JSONResponse(db.fine_summary(), headers=NO_CACHE)

    async def fine_update(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body expected"}, status_code=400)
        status = str(body.get("status", "")).upper()
        if status not in ("PAID", "CANCELLED", "ISSUED", "REVIEW"):
            return JSONResponse({"error": "status must be PAID, CANCELLED, ISSUED or REVIEW"}, status_code=400)
        ok = db.set_fine_status(request.path_params["no"], status, body.get("plate"))
        return JSONResponse({"ok": ok}, status_code=200 if ok else 404)

    routes = [
        Route("/", index),
        Route("/frame/{name}.jpg", frame),
        Route("/api/state", state),
        Route("/api/events", events),
        Route("/events/stream", stream),
        Route("/api/emergency/{approach}", emergency, methods=["POST"]),
        Route("/api/sim/ambulance/{approach}", sim_ambulance, methods=["POST"]),
        Route("/api/sim/inject/{kind}", sim_inject, methods=["POST"]),
        Route("/api/fines", fines),
        Route("/api/fines/summary", fines_summary),
        Route("/api/fines/{no}", fine_update, methods=["POST"]),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
        Mount("/evidence", StaticFiles(directory=evidence_dir), name="evidence"),
    ]
    return Starlette(routes=routes)
