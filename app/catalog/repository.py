"""Catalogue repository: the single source of truth for what the game engine supports.

Every other module (schemas, validator, retrieval, prompts) reads from here, so a
rule or an asset is defined in exactly one place: catalog.json.
"""

from __future__ import annotations

import json
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).parent / "catalog.json"

KINDS = ("templates", "environments", "assets", "rewards")


class Catalog:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def from_file(cls, path: Path = CATALOG_PATH) -> Catalog:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    # ---- raw access ------------------------------------------------------
    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    @property
    def version(self) -> str:
        return self._data["catalog_version"]

    @property
    def rules(self) -> dict[str, Any]:
        return self._data["rules"]

    @property
    def defaults(self) -> dict[str, Any]:
        return self._data["defaults"]

    @property
    def unsupported_mechanics(self) -> list[dict[str, Any]]:
        return self._data.get("unsupported_mechanics", [])

    def items(self, kind: str) -> list[dict[str, Any]]:
        return self._data[kind]

    # ---- indexed access --------------------------------------------------
    @cached_property
    def _index(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {kind: {i["id"]: i for i in self._data[kind]} for kind in KINDS}

    def by_id(self, kind: str) -> dict[str, dict[str, Any]]:
        return self._index[kind]

    def get(self, kind: str, item_id: str) -> dict[str, Any] | None:
        return self._index[kind].get(item_id)

    def assets_in(self, environment_id: str) -> list[dict[str, Any]]:
        return [a for a in self._data["assets"] if environment_id in a["environments"]]

    # ---- integrity -------------------------------------------------------
    def integrity_errors(self) -> list[str]:
        """Static checks on the catalogue itself. Must be empty (tested in CI)."""
        errors: list[str] = []
        for kind in KINDS:
            ids = [i["id"] for i in self._data[kind]]
            if len(ids) != len(set(ids)):
                errors.append(f"duplicate ids in {kind}")
        env_ids = set(self.by_id("environments"))
        bands = self.rules["score_by_difficulty"]
        for t in self._data["templates"]:
            for e in t["compatible_environments"]:
                if e not in env_ids:
                    errors.append(f"template {t['id']} references unknown environment {e}")
        for a in self._data["assets"]:
            for e in a["environments"]:
                if e not in env_ids:
                    errors.append(f"asset {a['id']} references unknown environment {e}")
            if a["difficulty"] not in bands:
                errors.append(f"asset {a['id']} has unknown difficulty {a['difficulty']}")
        d = self.defaults
        if d["game_template"] not in self.by_id("templates"):
            errors.append("default template does not exist")
        if d["environment"] not in env_ids:
            errors.append("default environment does not exist")
        if d["reward_id"] not in self.by_id("rewards"):
            errors.append("default reward does not exist")
        return errors


@lru_cache
def get_catalog() -> Catalog:
    return Catalog.from_file()
