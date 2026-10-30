"""Hybrid retriever: dense similarity + exact keyword bonus, over the catalogue.

score = cosine(prompt, entry) + KEYWORD_BONUS * [prompt contains entry name / tag]

Embeddings catch paraphrases ("messy kid's room" ~ Bedroom); keywords catch short
exact words that embeddings tend to blur ("key", "park").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.catalog.repository import Catalog
from app.services.retrieval.embedders import Embedder
from app.services.text import contains_term, word_set

SEARCHABLE = ("templates", "environments", "assets")


@dataclass(frozen=True)
class Hit:
    item: dict[str, Any]
    score: float
    semantic: float
    keyword_match: bool

    @property
    def id(self) -> str:
        return self.item["id"]


@dataclass(frozen=True)
class RetrievalResult:
    templates: list[Hit]
    environments: list[Hit]
    assets: list[Hit]

    @property
    def best_template(self) -> Hit:
        return self.templates[0]

    @property
    def best_environment(self) -> Hit:
        return self.environments[0]


def entry_text(item: dict[str, Any]) -> str:
    tags = ", ".join(item.get("tags", []))
    return f"{item['name']}. {item.get('description', '')} Keywords: {tags}"


class Retriever:
    def __init__(self, catalog: Catalog, embedder: Embedder, keyword_bonus: float = 0.2):
        self.catalog = catalog
        self.embedder = embedder
        self.keyword_bonus = keyword_bonus
        corpus = [entry_text(i) for kind in SEARCHABLE for i in catalog.items(kind)]
        embedder.fit(corpus)
        self._index: dict[str, tuple[list[dict], np.ndarray]] = {
            kind: (catalog.items(kind), embedder.encode([entry_text(i) for i in catalog.items(kind)]))
            for kind in SEARCHABLE
        }

    @property
    def backend(self) -> str:
        return self.embedder.name

    def _keyword_hit(self, prompt: str, ws: set[str], item: dict) -> bool:
        terms = [item["name"], item["id"].replace("_", " "), *item.get("tags", [])]
        return any(contains_term(prompt, t, ws) for t in terms)

    def search(self, prompt: str, kind: str, top_k: int | None = None,
               query_vec: np.ndarray | None = None) -> list[Hit]:
        items, embs = self._index[kind]
        q = query_vec if query_vec is not None else self.embedder.encode([prompt])[0]
        sims = embs @ q
        ws = word_set(prompt)
        hits = []
        for item, sim in zip(items, sims):
            kw = self._keyword_hit(prompt, ws, item)
            score = float(sim) + (self.keyword_bonus if kw else 0.0)
            hits.append(Hit(item=item, score=round(score, 3), semantic=round(float(sim), 3), keyword_match=kw))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k] if top_k else hits

    def retrieve(self, prompt: str) -> RetrievalResult:
        q = self.embedder.encode([prompt])[0]  # encode once, reuse for every kind
        return RetrievalResult(
            templates=self.search(prompt, "templates", query_vec=q),
            environments=self.search(prompt, "environments", query_vec=q),
            assets=self.search(prompt, "assets", query_vec=q),
        )
