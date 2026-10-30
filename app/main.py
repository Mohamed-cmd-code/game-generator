"""FastAPI entry point:  uvicorn app.main:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.container import Container, build_container
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.exceptions import LLMConfigurationError, LLMUnavailableError, NotFoundError
from app.core.logging import setup_logging


def create_app(container: Container | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.container is None:  # build heavy services once, at startup
            app.state.container = build_container(settings)
        yield

    app = FastAPI(
        title="AI Game Generator POC",
        version="1.0.0",
        description="Prompt -> RAG -> LLM (Groq) -> Validation -> Unity-ready JSON",
        lifespan=lifespan,
    )
    app.state.container = container
    app.include_router(router)

    @app.exception_handler(LLMUnavailableError)
    async def _llm_down(_: Request, exc: LLMUnavailableError):
        return JSONResponse(status_code=503, content={"detail": "LLM provider unavailable", "error": str(exc)})

    @app.exception_handler(LLMConfigurationError)
    async def _llm_unconfigured(_: Request, exc: LLMConfigurationError):
        return JSONResponse(status_code=503, content={"detail": "LLM not configured", "error": str(exc)})

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    return app


app = create_app()
