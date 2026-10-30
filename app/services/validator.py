"""Validation layer 2 and 3.

Layer 1 (app/domain): Pydantic shapes, enums and ranges.
Layer 2 (here): business rules that involve several fields and the catalogue.
Layer 3 (here): intent consistency - did the answer honour what the user explicitly
asked, or explicitly report the change?

Every message says what is wrong AND what is allowed, because the messages are sent
back to the LLM in the repair loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from app.catalog.repository import Catalog
from app.domain.game_config import GameConfig
from app.domain.llm_output import LLMOutput
from app.services.request_parser import ExplicitConstraints

RANK = {"easy": 0, "medium": 1, "hard": 2}


# --------------------------------------------------------------------------- layer 2
def validate_business_rules(cfg: GameConfig, catalog: Catalog) -> list[str]:
    errors: list[str] = []
    templates, assets = catalog.by_id("templates"), catalog.by_id("assets")
    tpl = templates[cfg.game_template]
    bands = catalog.rules["score_by_difficulty"]

    # compatibility: template <-> environment <-> win condition
    if cfg.environment not in tpl["compatible_environments"]:
        errors.append(
            f"template '{tpl['id']}' is not available in '{cfg.environment}'. "
            f"Allowed environments: {tpl['compatible_environments']}"
        )
    if cfg.win_condition.type != tpl["win_condition"]:
        errors.append(
            f"win_condition.type must be '{tpl['win_condition']}' for template '{tpl['id']}'"
        )

    # assets: must exist in this environment, no duplicates
    available = [a["id"] for a in catalog.assets_in(cfg.environment)]
    for obj in cfg.objects:
        if obj.asset_id not in available:
            errors.append(
                f"asset '{obj.asset_id}' does not exist in '{cfg.environment}'. "
                f"Assets available there: {available}"
            )
    ids = [o.asset_id for o in cfg.objects]
    if len(ids) != len(set(ids)):
        errors.append("objects contains duplicate asset_id values; each asset can appear once")
    if len(ids) > len(available):
        errors.append(
            f"'{cfg.environment}' only has {len(available)} assets; objects can hold at most that many"
        )

    # scoring: each score inside its difficulty band, and monotonic with difficulty
    for obj in cfg.objects:
        diff = assets[obj.asset_id]["difficulty"]
        lo, hi = bands[diff]
        if not lo <= obj.score <= hi:
            errors.append(
                f"asset '{obj.asset_id}' is {diff}: its score must be between {lo} and {hi} (got {obj.score})"
            )
    ordered = sorted(cfg.objects, key=lambda o: RANK[assets[o.asset_id]["difficulty"]])
    for easier, harder in pairwise(ordered):
        if (RANK[assets[harder.asset_id]["difficulty"]] > RANK[assets[easier.asset_id]["difficulty"]]
                and harder.score <= easier.score):
            errors.append(
                f"harder object '{harder.asset_id}' must score more than easier object '{easier.asset_id}'"
            )

    # quiz questions only for quiz_hunt, and mandatory there
    if cfg.game_template == "quiz_hunt":
        missing = [o.asset_id for o in cfg.objects if o.question is None]
        if missing:
            errors.append(f"quiz_hunt requires a question on every object; missing for {missing}")
    else:
        extra = [o.asset_id for o in cfg.objects if o.question is not None]
        if extra:
            errors.append(f"question must be null for template '{cfg.game_template}' (set on {extra})")

    # reward
    if cfg.reward and catalog.get("rewards", cfg.reward.reward_id)["type"] != cfg.reward.type:
        errors.append("reward.type does not match the catalogue reward type")
    return errors


# --------------------------------------------------------------------------- layer 3
@dataclass(frozen=True)
class Targets:
    """Values the answer must contain: explicit user values win, then profile targets."""

    explicit: ExplicitConstraints
    personalised: dict[str, int] = field(default_factory=dict)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def validate_intent(out: LLMOutput, targets: Targets, catalog: Catalog) -> list[str]:
    if out.config is None:
        return []
    cfg, rules, exp = out.config, catalog.rules, targets.explicit
    errors: list[str] = []
    adapted_needed: list[str] = []

    # duration
    dr = rules["duration_seconds"]
    if exp.duration_seconds is not None:
        expected = _clamp(exp.duration_seconds, dr["min"], dr["max"])
        if expected != exp.duration_seconds:
            adapted_needed.append(f"duration {exp.duration_seconds}s -> {expected}s")
        if cfg.duration_seconds != expected:
            errors.append(f"the user asked for {exp.duration_seconds}s: duration_seconds must be {expected}")
    elif "duration_seconds" in targets.personalised and cfg.duration_seconds != targets.personalised["duration_seconds"]:
        errors.append(
            f"duration_seconds must be {targets.personalised['duration_seconds']} (player-profile adjustment)"
        )

    # object count (capped by what the environment really has)
    cap = min(rules["hidden_objects"]["max"], len(catalog.assets_in(cfg.environment)))
    if exp.object_count is not None:
        expected = _clamp(exp.object_count, rules["hidden_objects"]["min"], cap)
        if expected != exp.object_count:
            adapted_needed.append(f"{exp.object_count} objects -> {expected}")
        if len(cfg.objects) != expected:
            errors.append(
                f"the user asked for {exp.object_count} objects: objects must contain exactly {expected} "
                f"(max {cap} available in '{cfg.environment}')"
            )
    elif "object_count" in targets.personalised:
        expected = _clamp(targets.personalised["object_count"], 1, cap)
        if len(cfg.objects) != expected:
            errors.append(f"objects must contain exactly {expected} items (player-profile adjustment)")

    # hints
    hr = rules["hints"]
    if exp.hints is not None:
        expected = _clamp(exp.hints, hr["min"], hr["max"])
        if expected != exp.hints:
            adapted_needed.append(f"{exp.hints} hints -> {expected}")
        if cfg.hints.max_hints != expected:
            errors.append(
                f"the user asked for {exp.hints} hints: hints.max_hints must be {expected} "
                f"and hints.enabled must be {str(expected > 0).lower()}"
            )
    elif "max_hints" in targets.personalised and cfg.hints.max_hints != targets.personalised["max_hints"]:
        errors.append(f"hints.max_hints must be {targets.personalised['max_hints']} (player-profile adjustment)")

    if adapted_needed and out.status != "adapted":
        errors.append(
            "status must be 'adapted' and each change explained in adaptations, because these "
            f"requested values are outside the allowed ranges: {adapted_needed}"
        )
    return errors


def validate_output(out: LLMOutput, targets: Targets, catalog: Catalog) -> list[str]:
    if out.config is None:
        return []
    errors = validate_business_rules(out.config, catalog)
    return errors + validate_intent(out, targets, catalog)
