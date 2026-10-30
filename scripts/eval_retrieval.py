"""Measure retrieval quality and calibrate the relevance thresholds.

    python -m scripts.eval_retrieval                 # backend from .env (auto)
    python -m scripts.eval_retrieval --backend tfidf

Prints top-1 accuracy for labelled prompts and the score gap between in-scope and
off-topic prompts; pick a threshold inside that gap. Paste the table in the README.
"""

from __future__ import annotations

import argparse

from app.catalog.repository import get_catalog
from app.core.config import get_settings
from app.services.retrieval.embedders import build_embedder
from app.services.retrieval.retriever import Retriever

# (prompt, expected template or None, expected environment or None)
IN_SCOPE = [
    ("Create a game in a messy bedroom, find 5 objects", "hidden_objects", "bedroom"),
    ("a game in my kid's room", None, "bedroom"),
    ("Hidden objects in the park: find a guitar, a cat and a key", "hidden_objects", "park"),
    ("Treasure hunt in the V-Shop, hard, 8 objects", "treasure_hunt", "v_shop"),
    ("a quiz in the store, 30 seconds, no hints", "quiz_hunt", "v_shop"),
    ("follow clues around the playground to find the treasure", "treasure_hunt", "park"),
    ("answer trivia questions about things at home", "quiz_hunt", "bedroom"),
    ("search for my lost toys outside in the garden", "hidden_objects", "park"),
    ("spot the hidden items on the shop shelves", "hidden_objects", "v_shop"),
    ("an educational question game in the bedroom", "quiz_hunt", "bedroom"),
]
OFF_TOPIC = [
    "write me a poem about the ocean",
    "what is the weather in Paris tomorrow",
    "bake a chocolate cake recipe",
    "explain quantum physics",
    "translate hello into Spanish",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, choices=["auto", "sentence-transformers", "tfidf"])
    args = ap.parse_args()
    s = get_settings()
    embedder = build_embedder(args.backend or s.embedding_backend, s.embedding_model)
    r = Retriever(get_catalog(), embedder, keyword_bonus=s.keyword_bonus)

    print(f"backend: {r.backend}  (default threshold {embedder.default_threshold})\n")
    print(f"| {'prompt':58} | top template (score) | top environment (score) | ok |")
    print(f"|{'-' * 60}|---|---|---|")
    correct = total = 0
    in_scores = []
    for prompt, exp_t, exp_e in IN_SCOPE:
        res = r.retrieve(prompt)
        t, e = res.best_template, res.best_environment
        good = (exp_t in (None, t.id)) and (exp_e in (None, e.id))
        correct += good
        total += 1
        in_scores.append(max(t.score, e.score))
        print(f"| {prompt[:58]:58} | {t.id} ({t.score:.2f}) | {e.id} ({e.score:.2f}) | {'yes' if good else 'NO'} |")
    off_scores = []
    for prompt in OFF_TOPIC:
        res = r.retrieve(prompt)
        t, e = res.best_template, res.best_environment
        off_scores.append(max(t.score, e.score))
        print(f"| {prompt[:58]:58} | {t.id} ({t.score:.2f}) | {e.id} ({e.score:.2f}) | off-topic |")

    print(f"\ntop-1 accuracy (in scope): {correct}/{total}")
    print(f"lowest in-scope max score : {min(in_scores):.3f}")
    print(f"highest off-topic score   : {max(off_scores):.3f}")
    print(f"-> choose a threshold between {max(off_scores):.2f} and {min(in_scores):.2f}")


if __name__ == "__main__":
    main()
