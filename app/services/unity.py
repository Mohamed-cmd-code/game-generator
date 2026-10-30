"""Deterministic GameConfig -> UnityGameConfig resolution (no LLM involved).

Resolves catalogue ids into scene / prefab paths so the Unity client can run:
Load Environment (scene_path) -> Load Assets (prefab_path) -> Apply Rules -> START GAME
"""

from __future__ import annotations

from app.catalog.repository import Catalog
from app.domain.api_models import UnityEnvironment, UnityGameConfig, UnityObject
from app.domain.game_config import GameConfig


def to_unity(cfg: GameConfig, game_id: str, catalog: Catalog) -> UnityGameConfig:
    env = catalog.get("environments", cfg.environment)
    assets = catalog.by_id("assets")
    objects = [
        UnityObject(
            asset_id=o.asset_id,
            prefab_path=assets[o.asset_id]["prefab_path"],
            difficulty=assets[o.asset_id]["difficulty"],
            score=o.score,
            order=i,
            question=o.question,
        )
        for i, o in enumerate(cfg.objects, start=1)
    ]
    return UnityGameConfig(
        schema_version=cfg.schema_version,
        catalog_version=catalog.version,
        game_id=game_id,
        game_template=cfg.game_template,
        environment=UnityEnvironment(id=env["id"], scene_path=env["scene_path"]),
        duration_seconds=cfg.duration_seconds,
        difficulty=cfg.difficulty,
        objects=objects,
        max_score=sum(o.score for o in objects),
        hints=cfg.hints,
        win_condition=cfg.win_condition,
        reward=cfg.reward,
    )
