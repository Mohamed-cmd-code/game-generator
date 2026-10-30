"""Shared fixtures. Tests never call Groq and never download a model:
a ScriptedLLM and the TF-IDF embedder make them free, fast and deterministic."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.container import build_container
from app.catalog.repository import get_catalog
from app.core.config import Settings
from app.main import create_app
from app.services.llm.fake import ScriptedLLM
from app.services.retrieval.embedders import TfidfEmbedder
from app.services.retrieval.retriever import Retriever
from app.services.store import GameStore

BEDROOM_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "game_template": "hidden_objects",
    "environment": "bedroom",
    "duration_seconds": 180,
    "difficulty": "medium",
    "objects": [
        {"asset_id": "red_shoe_01", "score": 20, "question": None},
        {"asset_id": "book_02", "score": 20, "question": None},
        {"asset_id": "teddy_bear_01", "score": 20, "question": None},
        {"asset_id": "headphones_01", "score": 40, "question": None},
        {"asset_id": "key_01", "score": 80, "question": None},
    ],
    "hints": {"enabled": True, "max_hints": 3},
    "win_condition": {"type": "find_all_objects"},
    "reward": {"type": "card", "reward_id": "demo_card_01"},
}

COMPATIBLE_PROMPT = (
    "Create a game in a messy bedroom. The player must find 5 objects in less than 3 minutes. "
    "Harder objects should give more points. Add 3 hints and a reward at the end."
)


def config(**over: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(BEDROOM_CONFIG)
    cfg.update(over)
    return cfg


def answer(status: str = "ok", cfg: dict | None = None, **over: Any) -> dict[str, Any]:
    base = {"status": status, "config": cfg if cfg is not None else config(), "message": "Here is your game.",
            "assumptions": [], "adaptations": [], "alternatives": []}
    base.update(over)
    return base


@pytest.fixture(scope="session")
def catalog():
    return get_catalog()


@pytest.fixture(scope="session")
def retriever(catalog):
    return Retriever(catalog, TfidfEmbedder())


@pytest.fixture
def store(tmp_path):
    return GameStore(tmp_path / "test.sqlite3")


@pytest.fixture
def make_container(retriever, store):
    def _make(answers):
        llm = ScriptedLLM(answers)
        settings = Settings(groq_api_key="", database_path=store.path, embedding_backend="tfidf")
        return build_container(settings, llm=llm, retriever=retriever, store=store), llm
    return _make


@pytest.fixture
def make_client(make_container):
    def _make(answers=()):
        container, llm = make_container(answers)
        return TestClient(create_app(container=container)), llm, container
    return _make
