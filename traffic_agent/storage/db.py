"""SQLite event log (zero setup, single file: data/traffic.db).

Tables:
  events  every event the system produced (signal plans, violations, crashes...)
  fines   e-challans (one row per FINE_ISSUED event)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from ..core.events import Event, EventType

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    approach TEXT,
    severity TEXT,
    message TEXT,
    data TEXT,
    evidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS fines (
    challan_no TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    plate TEXT,
    violation TEXT,
    approach TEXT,
    amount INTEGER,
    status TEXT,
    evidence TEXT,
    data TEXT
);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def on_event(self, ev: Event) -> None:
        """EventBus subscriber: persist events (skip noisy ones)."""
        if ev.type == EventType.SIGNAL_CHANGE and not ev.data.get("log"):
            return
        with self.lock:
            self.conn.execute(
                "INSERT INTO events (ts, type, approach, severity, message, data, evidence) VALUES (?,?,?,?,?,?,?)",
                (ev.ts, ev.type, ev.approach, ev.severity, ev.message,
                 json.dumps(ev.data, default=str), json.dumps(ev.evidence)))
            if ev.type == EventType.FINE_ISSUED:
                d = ev.data
                self.conn.execute(
                    "INSERT OR REPLACE INTO fines (challan_no, ts, plate, violation, approach, amount, status, "
                    "evidence, data) VALUES (?,?,?,?,?,?,?,?,?)",
                    (d["challan_no"], ev.ts, d.get("plate"), d.get("violation"), ev.approach,
                     int(d.get("amount", 0)), d.get("plate_status", "ISSUED"), json.dumps(ev.evidence),
                     json.dumps(d, default=str)))
            self.conn.commit()

    def recent_fines(self, limit: int = 100, plate: Optional[str] = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM fines", []
        if plate:
            q += " WHERE plate = ?"
            args.append(plate.upper().replace(" ", ""))
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"] or "[]")
            d["data"] = json.loads(d["data"] or "{}")
            out.append(d)
        return out

    def fine_summary(self) -> dict[str, Any]:
        with self.lock:
            rows = self.conn.execute("SELECT violation, status, COUNT(*) n, SUM(amount) rs FROM fines "
                                     "GROUP BY violation, status").fetchall()
        out: dict[str, Any] = {"count": 0, "amount": 0, "by_violation": {}, "review": 0}
        for r in rows:
            out["count"] += r["n"]
            out["amount"] += r["rs"] or 0
            out["by_violation"][r["violation"]] = out["by_violation"].get(r["violation"], 0) + r["n"]
            if r["status"] == "REVIEW":
                out["review"] += r["n"]
        return out

    def set_fine_status(self, challan_no: str, status: str, plate: Optional[str] = None) -> bool:
        with self.lock:
            if plate:
                cur = self.conn.execute("UPDATE fines SET status = ?, plate = ? WHERE challan_no = ?",
                                        (status, plate.upper().replace(" ", ""), challan_no))
            else:
                cur = self.conn.execute("UPDATE fines SET status = ? WHERE challan_no = ?", (status, challan_no))
            self.conn.commit()
        return cur.rowcount > 0

    def recent_events(self, limit: int = 100, types: Optional[list[str]] = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM events"
        args: list[Any] = []
        if types:
            q += " WHERE type IN (%s)" % ",".join("?" * len(types))
            args += types
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"] or "{}")
            d["evidence"] = json.loads(d["evidence"] or "[]")
            out.append(d)
        return out

    def counts_by_type(self, since_ts: float = 0.0) -> dict[str, int]:
        with self.lock:
            rows = self.conn.execute("SELECT type, COUNT(*) c FROM events WHERE ts >= ? GROUP BY type",
                                     (since_ts,)).fetchall()
        return {r["type"]: r["c"] for r in rows}

    def close(self) -> None:
        with self.lock:
            self.conn.close()
