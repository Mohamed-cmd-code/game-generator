"""Public HTTP contracts (request / response envelopes)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.domain.game_config import Difficulty, GameConfig, Hints, QuizQuestion, Reward, Strict, WinCondition
from app.domain.llm_output import Status


# --------------------------------------------------------------------------- request
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000, examples=[
        "Create a game in a messy bedroom. The player must find 5 objects in less than 3 minutes."
    ])
    player_id: str | None = Field(default=None, max_length=64, description="Enables personalisation (bonus).")


# --------------------------------------------------------------------------- unity
class UnityEnvironment(Strict):
    id: str
    scene_path: str


class UnityObject(Strict):
    asset_id: str
    prefab_path: str
    difficulty: Difficulty
    score: int
    order: int = Field(description="Spawn / collection order (used as the clue order in treasure_hunt).")
    question: QuizQuestion | None = None


class UnityGameConfig(Strict):
    """Fully resolved config: Unity needs no catalogue and no NLP to run it.
    game_config.json -> Load Environment -> Load Assets -> Apply Rules -> START GAME"""

    schema_version: str
    catalog_version: str
    game_id: str
    game_template: str
    environment: UnityEnvironment
    duration_seconds: int
    difficulty: Difficulty
    objects: list[UnityObject]
    max_score: int
    hints: Hints
    win_condition: WinCondition
    reward: Reward | None


# --------------------------------------------------------------------------- trace
class RetrievalHit(BaseModel):
    id: str
    score: float
    keyword_match: bool


class RetrievalTrace(BaseModel):
    backend: str
    templates: list[RetrievalHit]
    environments: list[RetrievalHit]
    assets: list[RetrievalHit]


class ResponseMeta(BaseModel):
    model: str | None = None
    attempts: int = 0
    rejected_by: Literal["mechanic_gate", "relevance_gate", "llm", "validation"] | None = None
    latency_ms: int = 0
    retrieval: RetrievalTrace | None = None


# --------------------------------------------------------------------------- response
class GenerateResponse(BaseModel):
    status: Status
    game_id: str | None = None
    message: str = ""
    config: GameConfig | None = None
    unity_config: UnityGameConfig | None = None
    assumptions: list[str] = Field(default_factory=list)
    adaptations: list[str] = Field(default_factory=list)
    personalization: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    @model_validator(mode="after")
    def _invariants(self) -> GenerateResponse:
        # The core guarantee of the service, enforced on the way out too.
        if self.status == "rejected" and (self.config or self.unity_config):
            raise ValueError("a rejected response never carries a config")
        if self.status != "rejected" and not (self.config and self.unity_config and self.game_id):
            raise ValueError("an accepted response always carries config, unity_config and game_id")
        return self


# --------------------------------------------------------------------------- bonus
class GameplayEvent(BaseModel):
    event: Literal["player_won", "player_failed", "player_quit"]
    game_id: str
    reason: Literal["time_expired", "gave_up", "wrong_answers", "other"] | None = None
    player_id: str | None = None
    time_used: int | None = Field(default=None, ge=0)
    objects_found: int | None = Field(default=None, ge=0)
    objects_total: int | None = Field(default=None, ge=0)
    hints_used: int | None = Field(default=None, ge=0)


class EventAck(BaseModel):
    stored: bool
    player_id: str | None
    profile: PlayerProfile | None = None


class Adjustment(BaseModel):
    field: Literal["duration_seconds", "object_count", "max_hints", "asset_difficulty"]
    target: int | str
    reason: str


class PlayerProfile(BaseModel):
    player_id: str
    games: int
    win_rate: float | None = None
    timeouts: int = 0
    avg_completion: float | None = None
    avg_hint_usage: float | None = None
    adjustments: list[Adjustment] = Field(default_factory=list)


EventAck.model_rebuild()
