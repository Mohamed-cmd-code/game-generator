import pytest
from pydantic import ValidationError

from app.domain.game_config import GameConfig, QuizQuestion
from app.domain.llm_output import LLMOutput
from tests.conftest import answer, config


# ---------------------------------------------------------------- catalogue
def test_catalog_integrity(catalog):
    assert catalog.integrity_errors() == []


def test_catalog_contains_assignment_items(catalog):
    assert {t["name"] for t in catalog.items("templates")} == {"Hidden Objects", "Treasure Hunt", "Quiz Hunt"}
    assert {e["name"] for e in catalog.items("environments")} == {"Bedroom", "V-Shop", "Park"}
    assert len(catalog.items("assets")) == 7
    r = catalog.rules
    assert (r["duration_seconds"], r["hidden_objects"], r["hints"], r["score_per_object"]) == (
        {"min": 30, "max": 180}, {"min": 1, "max": 10}, {"min": 0, "max": 5}, {"min": 10, "max": 100})


def test_every_environment_has_unity_paths(catalog):
    assert all(e["scene_path"] for e in catalog.items("environments"))
    assert all(a["prefab_path"] for a in catalog.items("assets"))


# ---------------------------------------------------------------- schema
def test_valid_config_parses():
    assert GameConfig.model_validate(config()).environment == "bedroom"


@pytest.mark.parametrize("field,value", [
    ("duration_seconds", 29), ("duration_seconds", 181), ("duration_seconds", "fast"),
    ("difficulty", "insane"), ("game_template", "racing"), ("environment", "tokyo"),
    ("schema_version", "2.0"),
])
def test_out_of_range_or_unknown_values_rejected(field, value):
    with pytest.raises(ValidationError):
        GameConfig.model_validate(config(**{field: value}))


def test_unknown_asset_rejected_by_schema():
    with pytest.raises(ValidationError):
        GameConfig.model_validate(config(objects=[{"asset_id": "guitar_01", "score": 20}]))


@pytest.mark.parametrize("score", [9, 101])
def test_score_range(score):
    with pytest.raises(ValidationError):
        GameConfig.model_validate(config(objects=[{"asset_id": "book_02", "score": score}]))


def test_object_count_bounds():
    with pytest.raises(ValidationError):
        GameConfig.model_validate(config(objects=[]))
    with pytest.raises(ValidationError):
        GameConfig.model_validate(config(objects=[{"asset_id": "book_02", "score": 20}] * 11))


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        GameConfig.model_validate(config(explosions=True))


@pytest.mark.parametrize("hints", [{"enabled": True, "max_hints": 0}, {"enabled": False, "max_hints": 2},
                                   {"enabled": True, "max_hints": 6}])
def test_inconsistent_hints_rejected(hints):
    with pytest.raises(ValidationError):
        GameConfig.model_validate(config(hints=hints))


def test_quiz_question_correct_index_must_exist():
    with pytest.raises(ValidationError):
        QuizQuestion.model_validate({"text": "What colour?", "choices": ["red", "blue"], "correct_index": 2})


def test_llm_output_status_invariants():
    with pytest.raises(ValidationError):
        LLMOutput.model_validate(answer(status="ok", cfg=None, config=None))
    with pytest.raises(ValidationError):
        LLMOutput.model_validate(answer(status="rejected"))  # config present on a rejection
    with pytest.raises(ValidationError):
        LLMOutput.model_validate(answer(status="adapted"))  # no adaptations listed
    with pytest.raises(ValidationError):
        LLMOutput.model_validate({"status": "rejected", "config": None, "message": "no"})  # no alternatives


def test_json_schema_exposes_catalog_enums():
    schema = GameConfig.model_json_schema()
    assert schema["properties"]["game_template"]["enum"] == ["hidden_objects", "treasure_hunt", "quiz_hunt"]
