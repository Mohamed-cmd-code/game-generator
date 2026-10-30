"""Bonus: adapt the next game from gameplay history (no model training).

Observe -> profile -> adjust -> observe again.
* Only the last N games count (recent skill, not lifetime).
* Cold start: fewer than MIN_GAMES -> no change.
* Small steps (+-25% time, +-1 object, +-1 hint), always clamped to catalogue rules.
* Target a 60-80% win rate: in that band nothing changes.
* Explicit user values always win (enforced by validator.validate_intent).
"""

from __future__ import annotations

from typing import Any

from app.catalog.repository import Catalog
from app.core.exceptions import NotFoundError
from app.domain.api_models import Adjustment, GameplayEvent, PlayerProfile
from app.services.store import GameStore

WINDOW = 5
MIN_GAMES = 2
LOW_WIN_RATE = 0.5
HIGH_WIN_RATE = 0.9


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class FeedbackService:
    def __init__(self, store: GameStore, catalog: Catalog):
        self.store = store
        self.catalog = catalog

    # ---- ingestion -------------------------------------------------------
    def record(self, event: GameplayEvent) -> str | None:
        game = self.store.get_game(event.game_id)
        if game is None:
            raise NotFoundError(f"unknown game_id '{event.game_id}'")
        player_id = event.player_id or game["player_id"]
        self.store.add_event(event.game_id, player_id, event.model_dump(exclude_none=True))
        return player_id

    # ---- profile ---------------------------------------------------------
    def profile(self, player_id: str) -> PlayerProfile:
        events = self.store.recent_events(player_id, WINDOW)
        if not events:
            return PlayerProfile(player_id=player_id, games=0)

        n = len(events)
        wins = sum(e["event"] == "player_won" for e in events)
        timeouts = sum(e.get("reason") == "time_expired" for e in events)
        completions = [e["objects_found"] / e["objects_total"] for e in events
                       if e.get("objects_total") and e.get("objects_found") is not None]
        hint_usage = [e["hints_used"] / e["config"]["hints"]["max_hints"] for e in events
                      if e.get("hints_used") is not None and e["config"]["hints"]["max_hints"] > 0]
        profile = PlayerProfile(
            player_id=player_id, games=n, win_rate=round(wins / n, 2), timeouts=timeouts,
            avg_completion=round(sum(completions) / len(completions), 2) if completions else None,
            avg_hint_usage=round(sum(hint_usage) / len(hint_usage), 2) if hint_usage else None,
        )
        profile.adjustments = self._adjustments(profile, events[-1]["config"])
        return profile

    def _adjustments(self, p: PlayerProfile, last_cfg: dict[str, Any]) -> list[Adjustment]:
        if p.games < MIN_GAMES or p.win_rate is None:
            return []
        r = self.catalog.rules
        dur, n_obj, hints = last_cfg["duration_seconds"], len(last_cfg["objects"]), last_cfg["hints"]["max_hints"]
        adj: list[Adjustment] = []

        if p.win_rate < LOW_WIN_RATE:  # struggling -> make it a bit easier
            if p.timeouts:
                new = _clamp(int(round(dur * 1.25 / 10) * 10), r["duration_seconds"]["min"], r["duration_seconds"]["max"])
                if new != dur:
                    adj.append(Adjustment(field="duration_seconds", target=new,
                                          reason=f"ran out of time in {p.timeouts} of the last {p.games} games"))
            if n_obj > r["hidden_objects"]["min"]:
                adj.append(Adjustment(field="object_count", target=n_obj - 1,
                                      reason=f"has a {p.win_rate:.0%} win rate, below the 60-80% target"))
            if hints < r["hints"]["max"] and (p.avg_hint_usage is None or p.avg_hint_usage >= 0.5):
                adj.append(Adjustment(field="max_hints", target=hints + 1,
                                      reason="relies on hints"))
            adj.append(Adjustment(field="asset_difficulty", target="easy",
                                  reason="prefer easier assets while the player is struggling"))
        elif p.win_rate > HIGH_WIN_RATE:  # dominating -> a bit harder
            new = _clamp(int(round(dur * 0.8 / 10) * 10), r["duration_seconds"]["min"], r["duration_seconds"]["max"])
            if new != dur:
                adj.append(Adjustment(field="duration_seconds", target=new,
                                      reason=f"won {p.win_rate:.0%} of recent games"))
            if n_obj < r["hidden_objects"]["max"]:
                adj.append(Adjustment(field="object_count", target=n_obj + 1,
                                      reason="finds everything easily"))
            adj.append(Adjustment(field="asset_difficulty", target="hard",
                                  reason="prefer harder assets to keep the game challenging"))
        return adj

    @staticmethod
    def targets(profile: PlayerProfile) -> dict[str, Any]:
        return {a.field: a.target for a in profile.adjustments}

    @staticmethod
    def describe(profile: PlayerProfile, applied_fields: set[str]) -> list[str]:
        labels = {"duration_seconds": "Duration set to {t} s", "object_count": "Object count set to {t}",
                  "max_hints": "Hints set to {t}", "asset_difficulty": "Preferring {t} assets"}
        return [f"{labels[a.field].format(t=a.target)} because the player {a.reason}"
                if a.field != "asset_difficulty" else f"{labels[a.field].format(t=a.target)} ({a.reason})"
                for a in profile.adjustments if a.field in applied_fields]
