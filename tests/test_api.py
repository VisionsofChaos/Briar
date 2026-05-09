"""Tests for the Briar FastAPI service."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_tts_returns_placeholder_response() -> None:
    response = client.post("/tts", json={"text": "Hello from Briar"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "placeholder",
        "message": "TTS generation is not implemented yet.",
        "text": "Hello from Briar",
        "audio_url": None,
    }


def test_tts_rejects_empty_text() -> None:
    response = client.post("/tts", json={"text": ""})

    assert response.status_code == 422


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
