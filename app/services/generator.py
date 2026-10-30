"""The generation pipeline.

prompt
  -> retrieval (hybrid scores)
  -> gate 1: unsupported mechanic detected and no supported template mentioned -> rejected
  -> gate 2: nothing in the catalogue is relevant                              -> rejected
  -> explicit constraints (regex) + player profile targets (bonus)
  -> LLM (JSON mode) on retrieved context only
  -> validation: schema (Pydantic) -> business rules -> intent consistency
  -> repair loop (errors sent back to the LLM) up to N attempts
  -> Unity resolution + persistence
Invariant: an invalid config is never returned. It is validated or the request is rejected.
"""

from __future__ import annotations

import logging
import time
import uuid

from pydantic import ValidationError

from app.catalog.repository import Catalog
from app.domain.api_models import GenerateResponse, ResponseMeta, RetrievalHit, RetrievalTrace
from app.domain.llm_output import LLMOutput
from app.services.feedback import FeedbackService
from app.services.guards import default_alternatives, detect_unsupported_mechanics
from app.services.llm.base import LLMClient, Message
from app.services.prompts import build_context, build_repair_message, build_system_prompt, build_user_message
from app.services.request_parser import parse_constraints
from app.services.retrieval.retriever import Hit, RetrievalResult, Retriever
from app.services.store import GameStore
from app.services.unity import to_unity
from app.services.validator import Targets, validate_output

log = logging.getLogger(__name__)


def _hits(hits: list[Hit], k: int) -> list[RetrievalHit]:
    return [RetrievalHit(id=h.id, score=h.score, keyword_match=h.keyword_match) for h in hits[:k]]


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors()[:15]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return out


class GameGenerator:
    def __init__(
        self,
        catalog: Catalog,
        retriever: Retriever,
        llm: LLMClient,
        store: GameStore,
        feedback: FeedbackService,
        template_threshold: float,
        environment_threshold: float,
        max_attempts: int = 3,
    ):
        self.catalog = catalog
        self.retriever = retriever
        self.llm = llm
        self.store = store
        self.feedback = feedback
        self.template_threshold = template_threshold
        self.environment_threshold = environment_threshold
        self.max_attempts = max_attempts
        self.system_prompt = build_system_prompt(catalog)

    # ------------------------------------------------------------------ public
    def generate(self, prompt: str, player_id: str | None = None) -> GenerateResponse:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(prompt)
        meta = ResponseMeta(model=self.llm.model, retrieval=self._trace(retrieval))

        def done(resp: GenerateResponse) -> GenerateResponse:
            resp.meta.latency_ms = int((time.perf_counter() - t0) * 1000)
            log.info("generate status=%s rejected_by=%s attempts=%s latency_ms=%s",
                     resp.status, resp.meta.rejected_by, resp.meta.attempts, resp.meta.latency_ms)
            return resp

        # gate 1: deterministic unsupported-mechanic detection
        mechanics = detect_unsupported_mechanics(prompt, self.catalog)
        template_named = any(h.keyword_match for h in retrieval.templates)
        if mechanics and not template_named:
            names = [m.label for m in mechanics]
            labels = names[0] + " is" if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] + " are"
            meta.rejected_by = "mechanic_gate"
            return done(GenerateResponse(
                status="rejected", meta=meta,
                message=(f"{labels} not currently supported. I can create a Hidden Objects, "
                         "Treasure Hunt or Quiz Hunt game in the Bedroom, V-Shop or Park instead."),
                alternatives=default_alternatives(self.catalog),
                errors=[f"unsupported mechanic: {m.label} ({', '.join(m.terms)})" for m in mechanics],
            ))

        # gate 2: relevance threshold on retrieval scores
        if (retrieval.best_template.score < self.template_threshold
                and retrieval.best_environment.score < self.environment_threshold):
            meta.rejected_by = "relevance_gate"
            return done(GenerateResponse(
                status="rejected", meta=meta,
                message="This request does not match any supported game type or environment.",
                alternatives=default_alternatives(self.catalog),
            ))

        # explicit constraints + personalisation (bonus)
        constraints = parse_constraints(prompt)
        profile, personalised = None, {}
        if player_id:
            profile = self.feedback.profile(player_id)
            explicit = constraints.as_dict()
            explicit_map = {"duration_seconds": "duration_seconds", "object_count": "object_count", "max_hints": "hints"}
            personalised = {k: v for k, v in self.feedback.targets(profile).items()
                            if explicit_map.get(k) not in explicit}
        targets = Targets(explicit=constraints,
                          personalised={k: v for k, v in personalised.items() if isinstance(v, int)})

        context = build_context(retrieval, self.catalog, self.environment_threshold,
                                constraints, mechanics, personalised or None)
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_user_message(context, prompt)},
        ]

        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            meta.attempts = attempt
            raw = self.llm.complete(messages)
            try:
                out = LLMOutput.model_validate_json(raw)
            except ValidationError as exc:
                errors = _format_pydantic_errors(exc)
            else:
                if out.status == "rejected":
                    meta.rejected_by = "llm"
                    return done(GenerateResponse(
                        status="rejected", message=out.message, meta=meta,
                        alternatives=out.alternatives or default_alternatives(self.catalog),
                    ))
                errors = validate_output(out, targets, self.catalog)
                if not errors:
                    return done(self._accept(out, prompt, player_id, profile, personalised, meta))
            log.info("attempt %s invalid: %s", attempt, errors)
            messages += [
                {"role": "assistant", "content": raw or "<empty answer>"},
                {"role": "user", "content": build_repair_message(errors)},
            ]

        meta.rejected_by = "validation"
        return done(GenerateResponse(
            status="rejected", meta=meta, errors=errors,
            message=f"Could not build a valid configuration after {self.max_attempts} attempts.",
            alternatives=default_alternatives(self.catalog),
        ))

    # ------------------------------------------------------------------ helpers
    def _accept(self, out: LLMOutput, prompt: str, player_id: str | None, profile, personalised, meta) -> GenerateResponse:
        assert out.config is not None
        game_id = f"game_{uuid.uuid4().hex[:12]}"
        unity = to_unity(out.config, game_id, self.catalog)
        self.store.save_game(game_id, player_id, prompt, unity.model_dump())
        return GenerateResponse(
            status=out.status, game_id=game_id, message=out.message,
            config=out.config, unity_config=unity,
            assumptions=out.assumptions, adaptations=out.adaptations,
            personalization=self.feedback.describe(profile, set(personalised)) if profile else [],
            meta=meta,
        )

    def _trace(self, r: RetrievalResult) -> RetrievalTrace:
        return RetrievalTrace(backend=self.retriever.backend, templates=_hits(r.templates, 3),
                              environments=_hits(r.environments, 3), assets=_hits(r.assets, 5))
