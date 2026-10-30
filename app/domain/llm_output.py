"""The exact JSON object the LLM must return. Kept separate from the API response:
the LLM decides the game, the server adds ids, Unity paths and metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from app.domain.game_config import GameConfig, Strict

Status = Literal["ok", "adapted", "rejected"]


class LLMOutput(Strict):
    status: Status
    config: GameConfig | None = None
    message: str = Field(default="", max_length=500)
    assumptions: list[str] = Field(default_factory=list)
    adaptations: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _status_matches_content(self) -> LLMOutput:
        if self.status in ("ok", "adapted") and self.config is None:
            raise ValueError("config is required when status is ok/adapted")
        if self.status == "rejected" and self.config is not None:
            raise ValueError("config must be null when status is rejected")
        if self.status == "rejected" and not self.alternatives:
            raise ValueError("a rejected answer must propose at least one alternative")
        if self.status == "adapted" and not self.adaptations:
            raise ValueError("status 'adapted' requires at least one entry in adaptations")
        return self
