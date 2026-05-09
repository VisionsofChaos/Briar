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


def test_home_serves_chat_frontend() -> None:
    response = main.home()
    html = response.body.decode("utf-8")

    assert response.status_code == 200
    assert "<title>Briar</title>" in html
    assert 'id="chat-log"' in html
    assert 'id="message-input"' in html
    assert "Send" in html
    assert "fetch('/chat'" in html
    assert "player.play()" in html


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


def test_chat_sends_lm_studio_reply_through_tts(tmp_path, monkeypatch) -> None:
    workflow_path = tmp_path / "qwen3_tts.json"
    workflow_path.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "Qwen3-TTS VoiceClone",
                    "inputs": {"target_text": "old text"},
                }
            }
        ),
        encoding="utf-8",
    )
    submitted_requests = []

    def fake_urlopen(request, timeout):
        submitted_requests.append((request, timeout))
        if request.full_url == "http://127.0.0.1:50123/v1/chat/completions":
            return FakeComfyUIResponse(
                {"choices": [{"message": {"content": "A warm little reply."}}]}
            )
        if request.full_url == "http://127.0.0.1:8188/prompt":
            return FakeComfyUIResponse({"prompt_id": "prompt-chat"})
        return FakeComfyUIResponse(
            {
                "prompt-chat": {
                    "outputs": {
                        "9": {
                            "files": [
                                {
                                    "filename": "chat.wav",
                                    "subfolder": "audio",
                                    "type": "output",
                                }
                            ]
                        }
                    }
                }
            }
        )

    monkeypatch.setattr(main, "WORKFLOW_PATH", workflow_path)
    monkeypatch.setattr(main, "urlopen", fake_urlopen)

    response = main.create_chat(main.ChatRequest(message="How are you?"))
    response_json = response.model_dump()

    assert response_json == {
        "prompt_id": "prompt-chat",
        "filename": "chat.wav",
        "subfolder": "audio",
        "type": "output",
        "audio_url": (
            "http://127.0.0.1:8188/view?filename=chat.wav&"
            "subfolder=audio&type=output"
        ),
        "message": "How are you?",
        "reply": "A warm little reply.",
    }

    lm_request, lm_timeout = submitted_requests[0]
    assert lm_request.full_url == "http://127.0.0.1:50123/v1/chat/completions"
    assert lm_request.get_method() == "POST"
    assert lm_timeout == 60

    lm_body = json.loads(lm_request.data.decode("utf-8"))
    assert lm_body["model"] == "qwen2.5-14b"
    assert lm_body["temperature"] == 0.8
    assert lm_body["messages"][0] == {
        "role": "system",
        "content": main.BRIAR_SYSTEM_PROMPT,
    }
    assert lm_body["messages"][1] == {"role": "user", "content": "How are you?"}

    prompt_request, _ = submitted_requests[1]
    prompt_body = json.loads(prompt_request.data.decode("utf-8"))
    assert prompt_body["prompt"]["1"]["inputs"]["target_text"] == "A warm little reply."


def test_chat_returns_502_for_bad_lm_studio_response(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "urlopen",
        lambda request, timeout: FakeComfyUIResponse({"choices": []}),
    )

    try:
        main.create_chat(main.ChatRequest(message="Hello"))
    except HTTPException as exc:
        assert exc.status_code == 502
        assert "LM Studio response did not include choices" in exc.detail
    else:
        raise AssertionError("Expected HTTPException")


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


def test_replace_qwen3_target_text_uses_voice_clone_node() -> None:
    workflow = {
        "1": {
            "class_type": "Qwen3TTSLoader",
            "inputs": {"text": "do not change"},
        },
        "2": {
            "class_type": "Qwen3-TTS VoiceClone",
            "inputs": {"target_text": "old text"},
        },
    }

    updated_workflow = main.replace_qwen3_target_text(workflow, "new text")

    assert updated_workflow["1"]["inputs"]["text"] == "do not change"
    assert updated_workflow["2"]["inputs"]["target_text"] == "new text"


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


def test_extract_audio_output_supports_files_field() -> None:
    history = {
        "prompt-files": {
            "outputs": {
                "7": {
                    "files": [
                        {
                            "filename": "from-files.wav",
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            }
        }
    }

    audio_output = main.extract_audio_output(history, "prompt-files")

    assert audio_output is not None
    assert audio_output.filename == "from-files.wav"
    assert audio_output.subfolder == ""
    assert audio_output.type == "output"


def test_extract_audio_output_supports_nested_filename_dict() -> None:
    history = {
        "prompt-nested": {
            "outputs": {
                "8": {
                    "metadata": {
                        "result": {
                            "filename": "nested.flac",
                            "subfolder": "audio",
                            "type": "output",
                        }
                    }
                }
            }
        }
    }

    audio_output = main.extract_audio_output(history, "prompt-nested")

    assert audio_output is not None
    assert audio_output.filename == "nested.flac"
    assert audio_output.subfolder == "audio"
    assert audio_output.type == "output"


def test_extract_audio_output_supports_any_nested_filename_list() -> None:
    history = {
        "prompt-any": {
            "outputs": {
                "9": {
                    "unexpected_key": [
                        {"ignored": "value"},
                        [
                            {
                                "filename": "fallback.wav",
                            }
                        ],
                    ]
                }
            }
        }
    }

    audio_output = main.extract_audio_output(history, "prompt-any")

    assert audio_output is not None
    assert audio_output.filename == "fallback.wav"
    assert audio_output.subfolder == ""
    assert audio_output.type == "output"


def test_extract_audio_output_detects_audio_subfolder_flac() -> None:
    history = {
        "prompt-flac": {
            "outputs": {
                "10": {
                    "files": [
                        {
                            "filename": "ComfyUI_00003_.flac",
                            "subfolder": "audio",
                            "type": "output",
                        }
                    ]
                }
            }
        }
    }

    audio_output = main.extract_audio_output(history, "prompt-flac")

    assert audio_output is not None
    assert audio_output.filename == "ComfyUI_00003_.flac"
    assert audio_output.subfolder == "audio"
    assert audio_output.type == "output"


def test_wait_for_audio_output_times_out_logs_output_keys(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        main,
        "urlopen",
        lambda request, timeout: FakeComfyUIResponse(
            {"prompt-timeout": {"outputs": {"42": {"images": []}}}}
        ),
    )
    monkeypatch.setattr(main.time, "sleep", lambda seconds: None)

    try:
        main.wait_for_audio_output("prompt-timeout", timeout_seconds=0)
    except HTTPException as exc:
        assert exc.status_code == 504
        assert "Timed out waiting for ComfyUI audio output" in exc.detail
        assert "prompt_id=prompt-timeout" in caplog.text
        assert "history_keys=['prompt-timeout']" in caplog.text
        assert "output_node_ids=['42']" in caplog.text
        assert "output_keys_by_node={'42': ['images']}" in caplog.text
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
