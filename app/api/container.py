"""Composition root: builds every service once and wires dependencies explicitly."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.catalog.repository import Catalog, get_catalog
from app.core.config import Settings
from app.core.exceptions import LLMConfigurationError
from app.services.feedback import FeedbackService
from app.services.generator import GameGenerator
from app.services.llm.base import LLMClient
from app.services.llm.groq_client import GroqLLM, UnconfiguredLLM
from app.services.retrieval.embedders import build_embedder
from app.services.retrieval.retriever import Retriever
from app.services.store import GameStore

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    catalog: Catalog
    retriever: Retriever
    llm: LLMClient
    store: GameStore
    feedback: FeedbackService
    generator: GameGenerator


def build_container(settings: Settings, llm: LLMClient | None = None,
                    retriever: Retriever | None = None, store: GameStore | None = None) -> Container:
    catalog = get_catalog()
    if errors := catalog.integrity_errors():
        raise RuntimeError(f"catalog.json is inconsistent: {errors}")

    if retriever is None:
        embedder = build_embedder(settings.embedding_backend, settings.embedding_model)
        retriever = Retriever(catalog, embedder, keyword_bonus=settings.keyword_bonus)
    log.info("retrieval backend: %s", retriever.backend)

    if llm is None:
        try:
            llm = GroqLLM(settings)
        except LLMConfigurationError as exc:
            log.warning("%s - /generate-game will answer 503 until it is set.", exc)
            llm = UnconfiguredLLM(str(exc))

    store = store or GameStore(settings.database_path)
    feedback = FeedbackService(store, catalog)
    default_t = retriever.embedder.default_threshold
    generator = GameGenerator(
        catalog=catalog, retriever=retriever, llm=llm, store=store, feedback=feedback,
        template_threshold=settings.template_threshold or default_t,
        environment_threshold=settings.environment_threshold or default_t,
        max_attempts=settings.max_generation_attempts,
    )
    return Container(settings, catalog, retriever, llm, store, feedback, generator)
