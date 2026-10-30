from app.domain.game_config import GameConfig
from app.domain.llm_output import LLMOutput
from app.services.request_parser import ExplicitConstraints, parse_constraints
from app.services.validator import Targets, validate_business_rules, validate_intent
from tests.conftest import answer, config


def rules(catalog, **over):
    return validate_business_rules(GameConfig.model_validate(config(**over)), catalog)


def test_valid_config_has_no_errors(catalog):
    assert rules(catalog) == []


def test_asset_not_in_environment(catalog):
    errs = rules(catalog, environment="v_shop")  # key_01 is not sold in the V-Shop
    assert any("key_01" in e and "does not exist in 'v_shop'" in e for e in errs)


def test_wrong_win_condition(catalog):
    errs = rules(catalog, win_condition={"type": "reach_treasure"})
    assert any("must be 'find_all_objects'" in e for e in errs)


def test_duplicate_assets(catalog):
    objs = [{"asset_id": "book_02", "score": 20}, {"asset_id": "book_02", "score": 20}]
    assert any("duplicate" in e for e in rules(catalog, objects=objs))


def test_score_outside_difficulty_band(catalog):
    objs = [{"asset_id": "key_01", "score": 30}]  # hard band is 60-100
    assert any("between 60 and 100" in e for e in rules(catalog, objects=objs))


def test_harder_object_must_score_more(catalog):
    objs = [{"asset_id": "book_02", "score": 30}, {"asset_id": "headphones_01", "score": 30}]
    assert any("must score more" in e for e in rules(catalog, objects=objs))


def test_quiz_hunt_requires_questions(catalog):
    errs = rules(catalog, game_template="quiz_hunt", win_condition={"type": "answer_all_questions"})
    assert any("requires a question" in e for e in errs)


def test_question_forbidden_outside_quiz(catalog):
    q = {"text": "What colour is it?", "choices": ["red", "blue"], "correct_index": 0}
    errs = rules(catalog, objects=[{"asset_id": "red_shoe_01", "score": 20, "question": q}])
    assert any("question must be null" in e for e in errs)


# ---------------------------------------------------------------- request parser
def test_parser_extracts_explicit_numbers():
    c = parse_constraints("Treasure hunt, 8 objects, 4 minutes, 6 hints")
    assert (c.duration_seconds, c.object_count, c.hints) == (240, 8, 6)


def test_parser_handles_words_and_no_hints():
    c = parse_constraints("find five hidden items in 90 seconds with no hints")
    assert (c.duration_seconds, c.object_count, c.hints) == (90, 5, 0)


def test_parser_ignores_unrelated_numbers():
    c = parse_constraints("Create a Formula 1 race in Tokyo with 40 cars")
    assert c.as_dict() == {}


# ---------------------------------------------------------------- intent consistency
def intent(catalog, out: dict, **explicit):
    return validate_intent(LLMOutput.model_validate(out), Targets(ExplicitConstraints(**explicit)), catalog)


def test_explicit_values_must_be_honoured(catalog):
    assert intent(catalog, answer(), duration_seconds=180, object_count=5, hints=3) == []
    errs = intent(catalog, answer(), duration_seconds=120)
    assert any("duration_seconds must be 120" in e for e in errs)


def test_out_of_range_request_must_be_reported_as_adapted(catalog):
    errs = intent(catalog, answer(status="ok"), duration_seconds=240)
    assert any("status must be 'adapted'" in e for e in errs)
    ok = answer(status="adapted", adaptations=["Duration reduced from 240 s to 180 s (maximum)."])
    assert intent(catalog, ok, duration_seconds=240) == []


def test_object_count_capped_by_environment(catalog):
    errs = intent(catalog, answer(status="adapted", adaptations=["x"]), object_count=8)
    assert any("exactly 7" in e for e in errs)  # bedroom has 7 assets


def test_personalised_targets_enforced_only_when_not_explicit(catalog):
    out = LLMOutput.model_validate(answer())
    t = Targets(ExplicitConstraints(), personalised={"duration_seconds": 150})
    assert any("player-profile" in e for e in validate_intent(out, t, catalog))
    t = Targets(ExplicitConstraints(duration_seconds=180), personalised={"duration_seconds": 150})
    assert validate_intent(out, t, catalog) == []
