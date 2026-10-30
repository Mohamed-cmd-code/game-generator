"""Application settings, loaded from environment variables / .env (12-factor style)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore", env_ignore_empty=True)

    # --- LLM (Groq) -------------------------------------------------------
    groq_api_key: str = Field(default="", description="Groq API key (gsk_...)")
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_reasoning_effort: str | None = None  # e.g. "low" for gpt-oss reasoning models; None disables
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2  # network-level retries handled by the Groq SDK

    # --- Pipeline ---------------------------------------------------------
    max_generation_attempts: int = Field(default=3, ge=1, le=5)  # 1 try + 2 repairs

    # --- Retrieval --------------------------------------------------------
    embedding_backend: Literal["auto", "sentence-transformers", "tfidf"] = "auto"
    embedding_model: str = "all-MiniLM-L6-v2"
    keyword_bonus: float = 0.2
    # None = use the backend's calibrated default (see retrieval/embedders.py)
    template_threshold: float | None = None
    environment_threshold: float | None = None

    # --- Storage (bonus: gameplay history) ---------------------------------
    database_path: Path = ROOT_DIR / "data" / "games.sqlite3"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
