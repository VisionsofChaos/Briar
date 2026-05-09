"""Tests for the Briar FastAPI service."""

import json

from fastapi import HTTPException

from app import main


class FakeComfyUIResponse:
    """Minimal context-manager response used to test ComfyUI requests."""

    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self) -> "FakeComfyUIResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


def test_tts_submits_workflow_waits_for_audio_and_returns_view_url(
    tmp_path, monkeypatch
) -> None:
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
    history_responses = [
        {"prompt-123": {"outputs": {}}},
        {
            "prompt-123": {
                "outputs": {
                    "9": {
                        "audio": [
                            {
                                "filename": "voice clip.wav",
                                "subfolder": "tts outputs",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        },
    ]

    def fake_urlopen(request, timeout):
        submitted_requests.append((request, timeout))
        if request.get_method() == "POST":
            return FakeComfyUIResponse({"prompt_id": "prompt-123"})
        return FakeComfyUIResponse(history_responses.pop(0))

    monkeypatch.setattr(main, "WORKFLOW_PATH", workflow_path)
    monkeypatch.setattr(main, "urlopen", fake_urlopen)
    monkeypatch.setattr(main.time, "sleep", lambda seconds: None)

    response = main.create_tts(main.TTSRequest(text="Hello from Briar"))
    response_json = response.model_dump()

    assert response_json == {
        "prompt_id": "prompt-123",
        "filename": "voice clip.wav",
        "subfolder": "tts outputs",
        "type": "output",
        "audio_url": (
            "http://127.0.0.1:8188/view?filename=voice+clip.wav&"
            "subfolder=tts+outputs&type=output"
        ),
    }

    prompt_request, prompt_timeout = submitted_requests[0]
    assert prompt_request.full_url == "http://127.0.0.1:8188/prompt"
    assert prompt_request.get_method() == "POST"
    assert prompt_timeout == 30

    body = json.loads(prompt_request.data.decode("utf-8"))
    assert body["prompt"]["1"]["inputs"]["target_text"] == "Hello from Briar"

    history_request, _ = submitted_requests[1]
    assert history_request.full_url == "http://127.0.0.1:8188/history/prompt-123"
    assert history_request.get_method() == "GET"


def test_tts_rejects_empty_text() -> None:
    try:
        main.TTSRequest(text="")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected request validation error")


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


def test_extract_audio_output_supports_audios_field() -> None:
    history = {
        "prompt-abc": {
            "outputs": {
                "5": {
                    "audios": [
                        {
                            "filename": "clip.wav",
                            "subfolder": "nested",
                            "type": "output",
                        }
                    ]
                }
            }
        }
    }

    audio_output = main.extract_audio_output(history, "prompt-abc")

    assert audio_output is not None
    assert audio_output.filename == "clip.wav"
    assert audio_output.subfolder == "nested"
    assert audio_output.type == "output"


def test_wait_for_audio_output_times_out(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "urlopen",
        lambda request, timeout: FakeComfyUIResponse({"prompt-timeout": {"outputs": {}}}),
    )
    monkeypatch.setattr(main.time, "sleep", lambda seconds: None)

    try:
        main.wait_for_audio_output("prompt-timeout", timeout_seconds=0)
    except HTTPException as exc:
        assert exc.status_code == 504
        assert "Timed out waiting for ComfyUI audio output" in exc.detail
    else:
        raise AssertionError("Expected HTTPException")


def test_submit_prompt_requires_prompt_id(monkeypatch) -> None:
    monkeypatch.setattr(main, "urlopen", lambda request, timeout: FakeComfyUIResponse({"ok": True}))

    try:
        main.submit_prompt_to_comfyui({})
    except main.ComfyUIError as exc:
        assert "prompt_id" in str(exc)
    else:
        raise AssertionError("Expected ComfyUIError")


def test_health_returns_ok() -> None:
    assert main.health() == {"status": "ok"}
