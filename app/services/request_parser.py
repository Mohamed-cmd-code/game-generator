"""Deterministic extraction of explicit numeric constraints from the prompt.

Why: the LLM is good at language, code is good at numbers. Extracting
"4 minutes", "8 objects", "6 hints" with regexes lets the validator *prove* that
the LLM honoured (or explicitly adapted) what the user asked, instead of trusting it.
The parser is intentionally conservative: when in doubt it extracts nothing and the
LLM decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.text import NUMBER_WORDS, normalize

_NUM = r"\b(\d+(?:[.,]\d+)?|" + "|".join(sorted((w for w in NUMBER_WORDS if w != "a"), key=len, reverse=True)) + r")"

_DURATION_RE = re.compile(_NUM + r"\s*(minutes?|mins?|m|seconds?|secs?|s)\b")
_OBJECTS_RE = re.compile(
    _NUM + r"\s+(?:hidden\s+|small\s+|different\s+|secret\s+)?(?:objects?|items?|things|clues)\b"
)
_HINTS_RE = re.compile(_NUM + r"\s+hints?\b")
_NO_HINTS_RE = re.compile(r"\b(no|without|zero)\s+hints?\b")


def _to_number(token: str) -> float:
    token = token.replace(",", ".")
    return float(NUMBER_WORDS[token]) if token in NUMBER_WORDS else float(token)


@dataclass(frozen=True)
class ExplicitConstraints:
    duration_seconds: int | None = None
    object_count: int | None = None
    hints: int | None = None  # 0 means "no hints"

    def as_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def parse_constraints(prompt: str) -> ExplicitConstraints:
    text = normalize(prompt)

    duration = None
    if m := _DURATION_RE.search(text):
        value, unit = _to_number(m.group(1)), m.group(2)
        duration = round(value * 60) if unit.startswith("m") else round(value)

    count = None
    if m := _OBJECTS_RE.search(text):
        count = int(_to_number(m.group(1)))

    hints = None
    if _NO_HINTS_RE.search(text):
        hints = 0
    elif m := _HINTS_RE.search(text):
        hints = int(_to_number(m.group(1)))

    return ExplicitConstraints(duration_seconds=duration, object_count=count, hints=hints)
