"""Deterministic guard: detect mechanics the engine cannot run (racing, shooting...).

This runs before the LLM. It is cheap, explainable and cannot be prompt-injected.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.catalog.repository import Catalog
from app.services.text import contains_term, word_set


@dataclass(frozen=True)
class MechanicHit:
    label: str
    terms: tuple[str, ...]


def detect_unsupported_mechanics(prompt: str, catalog: Catalog) -> list[MechanicHit]:
    ws = word_set(prompt)
    hits: list[MechanicHit] = []
    for group in catalog.unsupported_mechanics:
        found = tuple(t for t in group["terms"] if contains_term(prompt, t, ws))
        if found:
            hits.append(MechanicHit(label=group["label"], terms=found))
    return hits


def default_alternatives(catalog: Catalog) -> list[str]:
    envs = " / ".join(e["name"] for e in catalog.items("environments"))
    return [
        f"Hidden Objects in the {envs}: find items before the timer ends",
        "Treasure Hunt: follow clues through the Park or the V-Shop to reach a treasure",
        "Quiz Hunt: answer questions about objects found in the Bedroom",
    ]
