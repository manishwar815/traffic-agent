"""Ambulance / control-room alert over Telegram (free Bot API, no SDK needed).

Setup (once):
  1. In Telegram, talk to @BotFather -> /newbot -> copy the bot token.
  2. Send any message to your bot (or add it to the control-room group).
  3. Open https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id": ...}.
  4. export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...   (or put them in the YAML)

Without a token and chat id the alerter runs in DRY-RUN mode: the alert is
written to evidence/<day>/alert_*.json instead, so the demo still works offline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from ...config import TelegramCfg

API = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlerter:
    channel = "telegram"

    def __init__(self, cfg: TelegramCfg, evidence) -> None:
        self.cfg = cfg
        self.evidence = evidence
        self.token = cfg.bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = cfg.chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    @property
    def configured(self) -> bool:
        return bool(self.cfg.enabled and self.token and self.chat_id)

    def describe(self) -> str:
        if not self.cfg.enabled:
            return "alerts off"
        return "Telegram" if self.configured else "dry run (no Telegram token)"

    def send(self, text: str, photo: Optional[Path] = None, lat: Optional[float] = None,
             lon: Optional[float] = None, stem: str = "alert") -> dict:
        """Blocking; call it from a worker thread. Returns delivery info."""
        if not self.cfg.enabled:
            return {"delivered": False, "channel": "none", "error": "alerts disabled"}
        if not self.configured:
            full, rel = self.evidence.new_path(stem, "json")
            full.write_text(json.dumps({"ts": time.time(), "text": text, "photo": str(photo) if photo else None,
                                        "lat": lat, "lon": lon}, indent=2), encoding="utf-8")
            return {"delivered": False, "channel": "dry-run", "record": rel,
                    "error": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"}
        import requests
        try:
            if photo and Path(photo).exists():
                with open(photo, "rb") as f:
                    r = requests.post(API.format(token=self.token, method="sendPhoto"),
                                      data={"chat_id": self.chat_id, "caption": text[:1024]},
                                      files={"photo": f}, timeout=self.cfg.timeout_s)
            else:
                r = requests.post(API.format(token=self.token, method="sendMessage"),
                                  data={"chat_id": self.chat_id, "text": text}, timeout=self.cfg.timeout_s)
            ok = r.ok and r.json().get("ok", False)
            if ok and lat is not None and lon is not None:
                requests.post(API.format(token=self.token, method="sendLocation"),
                              data={"chat_id": self.chat_id, "latitude": lat, "longitude": lon},
                              timeout=self.cfg.timeout_s)
            return {"delivered": bool(ok), "channel": "telegram",
                    "error": None if ok else f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as exc:                       # no internet, bad token...
            return {"delivered": False, "channel": "telegram", "error": repr(exc)}
