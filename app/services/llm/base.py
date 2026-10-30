"""LLM port. The pipeline depends on this protocol only, never on a vendor SDK,
so the provider can be swapped (Groq, OpenAI, Ollama...) and tests use a fake."""

from __future__ import annotations

from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[Message]) -> str:
        """Return the raw text of the model answer (expected to be one JSON object).
        Raise LLMUnavailableError on transport/provider errors."""
        ...
