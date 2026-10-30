import json

import pytest

from app.core.exceptions import LLMUnavailableError
from tests.conftest import COMPATIBLE_PROMPT, answer, config


def test_compatible_request_ok(make_container):
    c, llm = make_container([answer()])
    res = c.generator.generate(COMPATIBLE_PROMPT)
    assert res.status == "ok" and res.game_id and res.meta.attempts == 1
    u = res.unity_config
    assert u.environment.scene_path == "Scenes/Bedroom_01"
    assert u.objects[-1].prefab_path == "Prefabs/Items/Key" and u.max_score == 180
    # the LLM saw retrieved context and the parsed constraints
    user_msg = llm.calls[0][1]["content"]
    assert '"explicit_constraints"' in user_msg and '"duration_seconds": 180' in user_msg


def test_unsupported_mechanic_rejected_without_calling_llm(make_container):
    c, llm = make_container([])
    res = c.generator.generate("Create a Formula 1 race in Tokyo with 40 cars and explosions.")
    assert res.status == "rejected" and res.config is None
    assert res.meta.rejected_by == "mechanic_gate" and res.message.startswith("Racing and") and "not currently supported" in res.message
    assert res.alternatives and llm.calls == []


def test_off_topic_rejected_by_relevance_gate(make_container):
    c, llm = make_container([])
    res = c.generator.generate("write me a poem about the ocean")
    assert res.status == "rejected" and res.meta.rejected_by == "relevance_gate" and llm.calls == []


def test_supported_template_with_unsupported_word_goes_to_llm(make_container):
    # "toy car" is an object request inside a supported game: the LLM adapts it
    adapted = answer(status="adapted", adaptations=["Toy car is not in the catalogue; replaced by Teddy Bear."])
    c, llm = make_container([adapted])
    res = c.generator.generate("Hidden objects in the bedroom: find a toy car")
    assert res.status == "adapted" and len(llm.calls) == 1
    assert "unsupported_terms_detected" in llm.calls[0][1]["content"]


def test_repair_loop_fixes_invalid_first_answer(make_container):
    bad = answer(cfg=config(environment="v_shop"))  # key_01 not in v_shop + win cond fine
    c, llm = make_container([bad, answer()])
    res = c.generator.generate(COMPATIBLE_PROMPT)
    assert res.status == "ok" and res.meta.attempts == 2
    repair_msg = llm.calls[1][-1]["content"]
    assert "does not exist in 'v_shop'" in repair_msg
    assert llm.calls[1][-2]["role"] == "assistant"  # previous answer is shown back


def test_broken_json_every_time_returns_clean_rejection(make_container):
    c, llm = make_container(["not json", "{oops", '{"status": "ok"}'])
    res = c.generator.generate(COMPATIBLE_PROMPT)
    assert res.status == "rejected" and res.config is None and res.unity_config is None
    assert res.meta.rejected_by == "validation" and res.errors and len(llm.calls) == 3


def test_hallucinated_asset_is_never_returned(make_container):
    hallucinated = json.loads(json.dumps(answer()))
    hallucinated["config"]["objects"][0]["asset_id"] = "guitar_01"
    c, _ = make_container([hallucinated] * 3)
    res = c.generator.generate(COMPATIBLE_PROMPT)
    assert res.status == "rejected" and res.config is None


def test_ok_status_for_out_of_range_request_triggers_repair(make_container):
    prompt = "hidden objects in the bedroom, 5 objects, 10 minutes, 3 hints"
    silent_clamp = answer(status="ok")
    honest = answer(status="adapted", adaptations=["Duration reduced from 600 s to 180 s (maximum)."])
    c, llm = make_container([silent_clamp, honest])
    res = c.generator.generate(prompt)
    assert res.status == "adapted" and res.meta.attempts == 2
    assert "status must be 'adapted'" in llm.calls[1][-1]["content"]


def test_llm_rejection_passes_through(make_container):
    rej = {"status": "rejected", "config": None, "message": "Shooting is not supported.",
           "alternatives": ["Hidden Objects in the Park"]}
    c, _ = make_container([rej])
    res = c.generator.generate("a hidden objects game where you find things in the park")
    assert res.status == "rejected" and res.meta.rejected_by == "llm"


def test_provider_error_propagates(make_container):
    c, _ = make_container([LLMUnavailableError("down")])
    with pytest.raises(LLMUnavailableError):
        c.generator.generate(COMPATIBLE_PROMPT)


def test_game_is_persisted(make_container):
    c, _ = make_container([answer()])
    res = c.generator.generate(COMPATIBLE_PROMPT, player_id="p_1")
    stored = c.store.get_game(res.game_id)
    assert stored["player_id"] == "p_1" and stored["config"]["game_id"] == res.game_id
