"""Bonus: gameplay events -> player profile -> adjusted next game."""

from tests.conftest import COMPATIBLE_PROMPT, answer, config


def _play(client, outcome, reason=None, player="p_42", **extra):
    gid = client.post("/generate-game", json={"prompt": COMPATIBLE_PROMPT, "player_id": player}).json()["game_id"]
    ev = {"event": outcome, "game_id": gid, **({"reason": reason} if reason else {}), **extra}
    assert client.post("/events", json=ev).status_code == 201
    return gid


def test_cold_start_has_no_adjustments(make_client):
    client, _, _ = make_client([answer()])
    _play(client, "player_failed", "time_expired")
    p = client.get("/players/p_42/profile").json()
    assert p["games"] == 1 and p["adjustments"] == []


def test_repeated_timeouts_make_next_game_easier(make_client):
    # 2 games played, both lost on time -> longer, fewer objects, more hints
    cfg = config(duration_seconds=120, hints={"enabled": True, "max_hints": 2})
    client, _, _ = make_client([answer(cfg=cfg), answer(cfg=cfg)])
    for _ in range(2):
        gid = client.post("/generate-game", json={"prompt": "hidden objects in the bedroom", "player_id": "p_42"}).json()["game_id"]
        client.post("/events", json={"event": "player_failed", "reason": "time_expired", "game_id": gid,
                                     "hints_used": 2, "objects_found": 3, "objects_total": 5})
    profile = client.get("/players/p_42/profile").json()
    targets = {a["field"]: a["target"] for a in profile["adjustments"]}
    assert profile["win_rate"] == 0 and profile["timeouts"] == 2
    assert targets == {"duration_seconds": 150, "object_count": 4, "max_hints": 3, "asset_difficulty": "easy"}


def test_profile_targets_are_sent_to_llm_and_enforced(make_client):
    cfg = config(duration_seconds=120, hints={"enabled": True, "max_hints": 2})
    personalised = config(duration_seconds=150, hints={"enabled": True, "max_hints": 3},
                          objects=config()["objects"][:4])
    client, llm, _ = make_client([answer(cfg=cfg), answer(cfg=cfg), answer(cfg=cfg), answer(cfg=personalised)])
    for _ in range(2):
        gid = client.post("/generate-game", json={"prompt": "hidden objects in the bedroom", "player_id": "p_9"}).json()["game_id"]
        client.post("/events", json={"event": "player_failed", "reason": "time_expired", "game_id": gid, "hints_used": 2})

    r = client.post("/generate-game", json={"prompt": "hidden objects in the bedroom", "player_id": "p_9"}).json()
    # 1st answer ignored the targets -> repaired; 2nd honoured them
    assert r["status"] == "ok" and r["meta"]["attempts"] == 2
    assert r["config"]["duration_seconds"] == 150
    assert any("ran out of time" in p for p in r["personalization"])
    assert '"personalised_targets"' in llm.calls[-1][1]["content"]


def test_explicit_user_value_beats_profile(make_client):
    cfg = config(duration_seconds=120, hints={"enabled": True, "max_hints": 2})
    client, _, _ = make_client([answer(cfg=cfg)] * 2 + [answer(cfg=config(duration_seconds=60, objects=config()["objects"][:4],
                                                                         hints={"enabled": True, "max_hints": 3}))])
    for _ in range(2):
        gid = client.post("/generate-game", json={"prompt": "hidden objects in the bedroom", "player_id": "p_7"}).json()["game_id"]
        client.post("/events", json={"event": "player_failed", "reason": "time_expired", "game_id": gid, "hints_used": 2})
    r = client.post("/generate-game", json={"prompt": "hidden objects in the bedroom, 60 seconds", "player_id": "p_7"}).json()
    assert r["status"] == "ok" and r["config"]["duration_seconds"] == 60
    assert not any("Duration" in p for p in r["personalization"])
