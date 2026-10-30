"""SQLite persistence for generated games and gameplay events (bonus feedback loop).

SQLite is enough for a POC (zero setup, file-based). The interface is small so it
can be replaced by Postgres later without touching the services.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,
    player_id   TEXT,
    prompt      TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     TEXT NOT NULL REFERENCES games(game_id),
    player_id   TEXT,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_player ON events(player_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GameStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- games -----------------------------------------------------------
    def save_game(self, game_id: str, player_id: str | None, prompt: str, config: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?)",
                (game_id, player_id, prompt, json.dumps(config), _now()),
            )

    def get_game(self, game_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if row is None:
            return None
        return {**dict(row), "config": json.loads(row["config_json"])}

    # ---- events ----------------------------------------------------------
    def add_event(self, game_id: str, player_id: str | None, payload: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO events (game_id, player_id, payload, created_at) VALUES (?, ?, ?, ?)",
                (game_id, player_id, json.dumps(payload), _now()),
            )

    def recent_events(self, player_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Most recent events of a player, oldest first, each joined with its game config."""
        with self._conn() as c:
            rows = c.execute(
                """SELECT e.payload, g.config_json FROM events e JOIN games g USING (game_id)
                   WHERE e.player_id = ? ORDER BY e.id DESC LIMIT ?""",
                (player_id, limit),
            ).fetchall()
        return [
            {**json.loads(r["payload"]), "config": json.loads(r["config_json"])} for r in reversed(rows)
        ]
