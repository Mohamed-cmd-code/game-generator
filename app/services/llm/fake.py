"""Deterministic fake LLM for tests: returns scripted answers and records calls."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from app.services.llm.base import Message


class ScriptedLLM:
    model = "scripted-fake"

    def __init__(self, answers: Iterable[dict[str, Any] | str | Exception]):
        self._answers = list(answers)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if not self._answers:
            raise AssertionError("ScriptedLLM called more times than scripted")
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)
