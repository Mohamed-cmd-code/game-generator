"""The GameConfig contract: what a Unity client receives.

Enumerations (template ids, environment ids, asset ids...) and numeric ranges are
generated from the catalogue at import time, so:
  * the JSON Schema shown to the LLM lists the only legal values (fewer hallucinations);
  * changing catalog.json is the only thing needed to add an asset or change a rule.

Pydantic covers single-field shape and range checks. Cross-field and catalogue
relationship checks (asset available in that environment, score bands...) live in
app/services/validator.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.catalog.repository import get_catalog

_CAT = get_catalog()
_R = _CAT.rules


def _ids(kind: str):
    return Literal[tuple(i["id"] for i in _CAT.items(kind))]  # type: ignore[valid-type]


TemplateId = _ids("templates")
EnvironmentId = _ids("environments")
AssetId = _ids("assets")
RewardId = _ids("rewards")
WinConditionType = Literal[tuple(sorted({t["win_condition"] for t in _CAT.items("templates")}))]  # type: ignore[valid-type]
Difficulty = Literal["easy", "medium", "hard"]

SCHEMA_VERSION = "1.0"


class Strict(BaseModel):
    """Unknown fields are an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


class QuizQuestion(Strict):
    text: str = Field(min_length=5, max_length=200)
    choices: list[str] = Field(
        min_length=_R["quiz_choices"]["min"], max_length=_R["quiz_choices"]["max"]
    )
    correct_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _index_in_range(self) -> QuizQuestion:
        if self.correct_index >= len(self.choices):
            raise ValueError("correct_index must point to one of the choices")
        if len({c.strip().lower() for c in self.choices}) != len(self.choices):
            raise ValueError("quiz choices must be distinct")
        return self


class ObjectItem(Strict):
    asset_id: AssetId
    score: int = Field(ge=_R["score_per_object"]["min"], le=_R["score_per_object"]["max"])
    question: QuizQuestion | None = Field(
        default=None, description="Required for quiz_hunt, must be null otherwise."
    )


class Hints(Strict):
    enabled: bool
    max_hints: int = Field(ge=_R["hints"]["min"], le=_R["hints"]["max"])

    @model_validator(mode="after")
    def _consistent(self) -> Hints:
        if not self.enabled and self.max_hints != 0:
            raise ValueError("max_hints must be 0 when hints are disabled")
        if self.enabled and self.max_hints == 0:
            raise ValueError("max_hints must be > 0 when hints are enabled")
        return self


class WinCondition(Strict):
    type: WinConditionType


class Reward(Strict):
    type: Literal["card"]
    reward_id: RewardId


class GameConfig(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    game_template: TemplateId
    environment: EnvironmentId
    duration_seconds: int = Field(
        ge=_R["duration_seconds"]["min"], le=_R["duration_seconds"]["max"]
    )
    difficulty: Difficulty
    objects: list[ObjectItem] = Field(
        min_length=_R["hidden_objects"]["min"], max_length=_R["hidden_objects"]["max"]
    )
    hints: Hints
    win_condition: WinCondition
    reward: Reward | None = None
