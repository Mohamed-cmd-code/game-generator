from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, status

from app.api.container import Container
from app.core.exceptions import NotFoundError
from app.domain.api_models import EventAck, GameplayEvent, GenerateRequest, GenerateResponse, PlayerProfile

router = APIRouter()


def get_container(request: Request) -> Container:
    return request.app.state.container


Deps = Annotated[Container, Depends(get_container)]


@router.get("/health", tags=["ops"])
def health(c: Deps) -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_model": c.llm.model,
        "llm_configured": c.llm.model != "unconfigured",
        "retrieval_backend": c.retriever.backend,
        "catalog_version": c.catalog.version,
    }


@router.get("/catalog", tags=["catalog"])
def catalog(c: Deps) -> dict[str, Any]:
    """The controlled knowledge base (handy for reviewers and Unity developers)."""
    return c.catalog.raw


@router.post(
    "/generate-game",
    response_model=GenerateResponse,
    response_model_exclude_none=False,
    tags=["generation"],
    responses={503: {"description": "LLM provider unavailable or not configured"}},
)
def generate_game(req: GenerateRequest, c: Deps) -> GenerateResponse:
    """Turn a natural-language request into a validated, Unity-ready game configuration.

    Always HTTP 200 for a processed request: `status` is `ok`, `adapted` or `rejected`.
    """
    return c.generator.generate(req.prompt, req.player_id)


@router.get("/games/{game_id}", tags=["generation"])
def get_game(game_id: str, c: Deps) -> dict[str, Any]:
    """Unity can fetch a generated config by id (game_config.json)."""
    game = c.store.get_game(game_id)
    if game is None:
        raise NotFoundError(f"unknown game_id '{game_id}'")
    return game["config"]


@router.post("/events", response_model=EventAck, status_code=status.HTTP_201_CREATED, tags=["feedback"])
def post_event(event: GameplayEvent, c: Deps) -> EventAck:
    """Ingest a gameplay event sent by Unity (bonus: feedback loop)."""
    player_id = c.feedback.record(event)
    return EventAck(stored=True, player_id=player_id,
                    profile=c.feedback.profile(player_id) if player_id else None)


@router.get("/players/{player_id}/profile", response_model=PlayerProfile, tags=["feedback"])
def player_profile(player_id: str, c: Deps) -> PlayerProfile:
    return c.feedback.profile(player_id)
