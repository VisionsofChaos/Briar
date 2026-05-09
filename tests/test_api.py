"""Tests for the Briar FastAPI service."""

import json
from io import BytesIO

from fastapi.testclient import TestClient

from app import main
from app.main import app


client = TestClient(app)


class FakeComfyUIResponse:
    """Minimal context-manager response used to test ComfyUI submission."""

    def __enter__(self) -> "FakeComfyUIResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"prompt_id":"prompt-123"}'


def test_tts_submits_workflow_and_returns_prompt_id(tmp_path, monkeypatch) -> None:
    workflow_path = tmp_path / "qwen3_tts.json"
    workflow_path.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "Qwen3-TTS VoiceClone",
                    "inputs": {
                        "target_text": "old text",
                        "ref_audio": "clip_015.wav",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    submitted_requests = []

    def fake_urlopen(request, timeout):
        submitted_requests.append((request, timeout))
        return FakeComfyUIResponse()

    monkeypatch.setattr(main, "WORKFLOW_PATH", workflow_path)
    monkeypatch.setattr(main, "urlopen", fake_urlopen)

    response = client.post("/tts", json={"text": "Hello from Briar"})

    assert response.status_code == 200
    assert response.json() == {"prompt_id": "prompt-123"}
    assert len(submitted_requests) == 1

    request, timeout = submitted_requests[0]
    assert request.full_url == "http://127.0.0.1:8188/prompt"
    assert request.get_method() == "POST"
    assert timeout == 30

    body = json.loads(request.data.decode("utf-8"))
    assert body["prompt"]["1"]["inputs"]["target_text"] == "Hello from Briar"


def test_tts_rejects_empty_text() -> None:
    response = client.post("/tts", json={"text": ""})

    assert response.status_code == 422


def test_replace_qwen3_target_text_supports_text_field() -> None:
    workflow = {
        "3": {
            "class_type": "Qwen3TTSVoiceClone",
            "inputs": {"text": "old text"},
        }
    }

    updated_workflow = main.replace_qwen3_target_text(workflow, "new text")

    assert updated_workflow["3"]["inputs"]["text"] == "new text"
    assert workflow["3"]["inputs"]["text"] == "old text"


def test_submit_prompt_requires_prompt_id(monkeypatch) -> None:
    class ResponseWithoutPromptId:
        def __enter__(self) -> "ResponseWithoutPromptId":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return BytesIO(b'{"ok":true}').read()

    monkeypatch.setattr(main, "urlopen", lambda request, timeout: ResponseWithoutPromptId())

    try:
        main.submit_prompt_to_comfyui({})
    except main.ComfyUIError as exc:
        assert "prompt_id" in str(exc)
    else:
        raise AssertionError("Expected ComfyUIError")


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
