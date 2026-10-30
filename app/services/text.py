"""Small text helpers shared by retrieval, guards and the request parser."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")

NUMBER_WORDS = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90,
}


def normalize(text: str) -> str:
    return text.lower().replace("’", "'")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def word_set(text: str) -> set[str]:
    ws = words(text)
    return set(ws) | {singular(w) for w in ws}


def contains_term(text: str, term: str, wordset: set[str] | None = None) -> bool:
    """Whole-word / whole-phrase match (so 'car' does not match 'card')."""
    term = normalize(term)
    if " " in term or "-" in term or "'" in term:
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalize(text)) is not None
    ws = wordset if wordset is not None else word_set(text)
    return term in ws
