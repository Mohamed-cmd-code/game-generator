"""Groq adapter (https://console.groq.com). Uses JSON mode + temperature 0."""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.core.exceptions import LLMConfigurationError, LLMUnavailableError
from app.services.llm.base import Message

log = logging.getLogger(__name__)


class GroqLLM:
    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise LLMConfigurationError("GROQ_API_KEY is not set (copy .env.example to .env)")
        import groq  # imported here so the rest of the app does not depend on the SDK

        self._groq = groq
        self.model = settings.groq_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._reasoning_effort = settings.llm_reasoning_effort
        self._client = groq.Groq(
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,  # retries 429 / 5xx with backoff
        )

    def complete(self, messages: list[Message]) -> str:
        try:
            extra = {"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else {}
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **extra,
            )
        except self._groq.BadRequestError as exc:
            # Groq returns 400 'json_validate_failed' when the model emits invalid JSON:
            # treat it as a bad answer so the repair loop can try again. Any other 400
            # (unsupported parameter, prompt too long...) is a configuration problem.
            if "json_validate_failed" not in str(exc):
                raise LLMUnavailableError(f"Groq rejected the request: {exc}") from exc
            log.warning("Groq rejected the generation as invalid JSON: %s", exc)
            return ""
        except self._groq.APIError as exc:  # connection, timeout, auth, rate limit, 5xx
            raise LLMUnavailableError(f"Groq API error: {exc.__class__.__name__}: {exc}") from exc
        usage = getattr(resp, "usage", None)
        if usage:
            log.info("groq model=%s prompt_tokens=%s completion_tokens=%s",
                     self.model, usage.prompt_tokens, usage.completion_tokens)
        return resp.choices[0].message.content or ""


class UnconfiguredLLM:
    """Placeholder used when no API key is set: the app still boots (health, catalog,
    events work) and /generate-game answers 503 with a clear message."""

    model = "unconfigured"

    def __init__(self, reason: str):
        self._reason = reason

    def complete(self, messages: list[Message]) -> str:
        raise LLMConfigurationError(self._reason)