import pytest

from app.services.guards import detect_unsupported_mechanics

THRESHOLD = 0.25  # TfidfEmbedder.default_threshold


@pytest.mark.parametrize("prompt,expected", [
    ("a game in a messy kid's room", "bedroom"),
    ("something in a store with shelves", "v_shop"),
    ("hide things in the garden outside", "park"),
    ("Treasure hunt in the V-Shop", "v_shop"),
])
def test_environment_top1(retriever, prompt, expected):
    assert retriever.retrieve(prompt).best_environment.id == expected


@pytest.mark.parametrize("prompt,expected", [
    ("find hidden objects", "hidden_objects"),
    ("follow the clues to the treasure", "treasure_hunt"),
    ("answer trivia questions", "quiz_hunt"),
])
def test_template_top1(retriever, prompt, expected):
    assert retriever.retrieve(prompt).best_template.id == expected


def test_keyword_bonus_on_exact_asset(retriever):
    hit = retriever.search("find the key", "assets", top_k=1)[0]
    assert hit.id == "key_01" and hit.keyword_match


def test_keyword_match_is_whole_word(retriever):
    # "card" must not trigger the "car" racing term, nor "keyboard" the key asset
    hits = {h.id: h for h in retriever.search("a keyboard and a card", "assets")}
    assert not hits["key_01"].keyword_match


@pytest.mark.parametrize("prompt", [
    "write me a poem about the ocean", "what is the weather in Paris tomorrow",
    "bake a chocolate cake recipe", "explain quantum physics",
])
def test_off_topic_prompts_score_below_threshold(retriever, prompt):
    r = retriever.retrieve(prompt)
    assert r.best_template.score < THRESHOLD and r.best_environment.score < THRESHOLD


def test_unsupported_mechanics_detected(catalog):
    labels = {h.label for h in detect_unsupported_mechanics(
        "Create a Formula 1 race in Tokyo with 40 cars and explosions.", catalog)}
    assert {"Racing", "Explosions / destruction"} <= labels
    assert detect_unsupported_mechanics("find a red card in the bedroom", catalog) == []
