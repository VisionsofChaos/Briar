"""FastAPI service entry point for Briar."""

import copy
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


COMFYUI_BASE_URL = "http://127.0.0.1:8188"
COMFYUI_PROMPT_URL = f"{COMFYUI_BASE_URL}/prompt"
DEFAULT_HISTORY_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 1
WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "workflows" / "qwen3_tts.json"
TARGET_TEXT_FIELDS = ("target_text", "text", "prompt")

app = FastAPI(
    title="Briar TTS Service",
    version="0.1.0",
    description="A small API surface for submitting text-to-speech jobs to ComfyUI.",
)


class TTSRequest(BaseModel):
    """Request body for text-to-speech generation."""

    text: str = Field(..., min_length=1, description="Text to synthesize.")


class TTSResponse(BaseModel):
    """Response returned after ComfyUI finishes generating audio."""

    prompt_id: str
    filename: str
    subfolder: str
    type: str
    audio_url: str


class AudioOutput(BaseModel):
    """Generated audio location returned by ComfyUI history."""

    filename: str
    subfolder: str = ""
    type: str = "output"


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI does not return the expected response."""


def load_workflow(workflow_path: Path | None = None) -> dict[str, Any]:
    """Load a ComfyUI API-format workflow from disk."""

    if workflow_path is None:
        workflow_path = WORKFLOW_PATH

    try:
        with workflow_path.open("r", encoding="utf-8") as workflow_file:
            workflow = json.load(workflow_file)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow file not found: {workflow_path}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow file is not valid JSON: {workflow_path}",
        ) from exc

    if not isinstance(workflow, dict):
        raise HTTPException(status_code=500, detail="Workflow JSON must be an object.")

    return workflow


def node_matches_qwen3_tts(node: dict[str, Any]) -> bool:
    """Return whether a ComfyUI node appears to be a Qwen3-TTS node."""

    metadata = node.get("_meta")
    title = metadata.get("title", "") if isinstance(metadata, dict) else ""
    searchable_values = (node.get("class_type", ""), node.get("title", ""), title)
    searchable_text = " ".join(str(value).lower() for value in searchable_values)
    compact_text = searchable_text.replace("-", "").replace("_", "").replace(" ", "")

    return "qwen3" in compact_text and "tts" in compact_text


def replace_qwen3_target_text(workflow: dict[str, Any], text: str) -> dict[str, Any]:
    """Return a workflow copy with the Qwen3-TTS target text replaced."""

    updated_workflow = copy.deepcopy(workflow)

    for node in updated_workflow.values():
        if not isinstance(node, dict) or not node_matches_qwen3_tts(node):
            continue

        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise HTTPException(
                status_code=500,
                detail="Qwen3-TTS node does not contain an inputs object.",
            )

        for field_name in TARGET_TEXT_FIELDS:
            if field_name in inputs:
                inputs[field_name] = text
                return updated_workflow

        raise HTTPException(
            status_code=500,
            detail=(
                "Qwen3-TTS node is missing a target text input field. "
                f"Expected one of: {', '.join(TARGET_TEXT_FIELDS)}."
            ),
        )

    raise HTTPException(
        status_code=500,
        detail="Could not find a Qwen3-TTS node in the workflow.",
    )


def read_json_from_comfyui(request: Request, timeout: int = 30) -> dict[str, Any]:
    """Send a request to ComfyUI and return a JSON object response."""

    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI returned HTTP {exc.code}: {detail}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to ComfyUI at {request.full_url}: {exc.reason}",
        ) from exc

    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ComfyUIError("ComfyUI returned a non-JSON response.") from exc

    if not isinstance(response_json, dict):
        raise ComfyUIError("ComfyUI returned JSON that was not an object.")

    return response_json


def submit_prompt_to_comfyui(
    workflow: dict[str, Any], prompt_url: str = COMFYUI_PROMPT_URL
) -> str:
    """Submit a ComfyUI workflow and return the accepted prompt id."""

    request_body = json.dumps({"prompt": workflow}).encode("utf-8")
    request = Request(
        prompt_url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response_json = read_json_from_comfyui(request)

    prompt_id = response_json.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ComfyUIError("ComfyUI response did not include a prompt_id.")

    return prompt_id


def extract_audio_output(history: dict[str, Any], prompt_id: str) -> AudioOutput | None:
    """Extract the first generated audio output from a ComfyUI history response."""

    prompt_history = history.get(prompt_id)
    if not isinstance(prompt_history, dict):
        return None

    outputs = prompt_history.get("outputs")
    if not isinstance(outputs, dict):
        return None

    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue

        audio_items = node_output.get("audio") or node_output.get("audios")
        if not isinstance(audio_items, list):
            continue

        for audio_item in audio_items:
            if not isinstance(audio_item, dict):
                continue

            filename = audio_item.get("filename")
            if not isinstance(filename, str) or not filename:
                continue

            subfolder = audio_item.get("subfolder", "")
            audio_type = audio_item.get("type", "output")
            return AudioOutput(
                filename=filename,
                subfolder=subfolder if isinstance(subfolder, str) else "",
                type=audio_type if isinstance(audio_type, str) and audio_type else "output",
            )

    return None


def wait_for_audio_output(
    prompt_id: str,
    timeout_seconds: int = DEFAULT_HISTORY_TIMEOUT_SECONDS,
    poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
    base_url: str = COMFYUI_BASE_URL,
) -> AudioOutput:
    """Poll ComfyUI history until an audio output appears or timeout expires."""

    history_url = f"{base_url}/history/{prompt_id}"
    deadline = time.monotonic() + timeout_seconds

    while True:
        request = Request(history_url, method="GET")
        history = read_json_from_comfyui(request)
        audio_output = extract_audio_output(history, prompt_id)
        if audio_output is not None:
            return audio_output

        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Timed out waiting for ComfyUI audio output "
                    f"for prompt_id {prompt_id} after {timeout_seconds} seconds."
                ),
            )

        time.sleep(poll_interval_seconds)


def build_audio_url(audio_output: AudioOutput, base_url: str = COMFYUI_BASE_URL) -> str:
    """Build the ComfyUI /view URL for a generated audio output."""

    query = urlencode(
        {
            "filename": audio_output.filename,
            "subfolder": audio_output.subfolder,
            "type": audio_output.type,
        }
    )
    return f"{base_url}/view?{query}"


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a simple health check response."""

    return {"status": "ok"}


@app.post("/tts", response_model=TTSResponse, tags=["tts"])
def create_tts(request: TTSRequest) -> TTSResponse:
    """Submit the Qwen3-TTS workflow to ComfyUI and return generated audio details."""

    workflow = load_workflow()
    updated_workflow = replace_qwen3_target_text(workflow, request.text)

    try:
        prompt_id = submit_prompt_to_comfyui(updated_workflow)
        audio_output = wait_for_audio_output(prompt_id)
    except ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TTSResponse(
        prompt_id=prompt_id,
        filename=audio_output.filename,
        subfolder=audio_output.subfolder,
        type=audio_output.type,
        audio_url=build_audio_url(audio_output),
    )
