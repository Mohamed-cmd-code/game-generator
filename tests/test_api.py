from app.core.exceptions import LLMUnavailableError
from tests.conftest import COMPATIBLE_PROMPT, answer


def test_health(make_client):
    client, _, _ = make_client()
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["retrieval_backend"] == "tfidf-char-ngrams"


def test_catalog_endpoint(make_client):
    client, _, _ = make_client()
    assert len(client.get("/catalog").json()["assets"]) == 7


def test_generate_ok(make_client):
    client, _, _ = make_client([answer()])
    r = client.post("/generate-game", json={"prompt": COMPATIBLE_PROMPT})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["unity_config"]["environment"]["scene_path"]
    assert client.get(f"/games/{body['game_id']}").json()["game_id"] == body["game_id"]


def test_rejected_is_http_200(make_client):
    client, _, _ = make_client()
    r = client.post("/generate-game", json={"prompt": "Create a Formula 1 race in Tokyo with 40 cars and explosions."})
    assert r.status_code == 200 and r.json()["status"] == "rejected" and r.json()["config"] is None


def test_empty_prompt_is_422(make_client):
    client, _, _ = make_client()
    assert client.post("/generate-game", json={"prompt": ""}).status_code == 422
    assert client.post("/generate-game", json={}).status_code == 422


def test_llm_down_is_503(make_client):
    client, _, _ = make_client([LLMUnavailableError("timeout")])
    r = client.post("/generate-game", json={"prompt": COMPATIBLE_PROMPT})
    assert r.status_code == 503 and r.json()["detail"] == "LLM provider unavailable"


def test_unknown_game_is_404(make_client):
    client, _, _ = make_client()
    assert client.get("/games/nope").status_code == 404
    r = client.post("/events", json={"event": "player_failed", "game_id": "nope"})
    assert r.status_code == 404
