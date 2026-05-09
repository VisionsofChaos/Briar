"""FastAPI service entry point for Briar."""

import copy
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


COMFYUI_PROMPT_URL = "http://127.0.0.1:8188/prompt"
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
    """Response returned after ComfyUI accepts a prompt."""

    prompt_id: str


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


def node_matches_qwen3_voice_clone(node: dict[str, Any]) -> bool:
    """Return whether a ComfyUI node appears to be a Qwen3-TTS VoiceClone node."""

    metadata = node.get("_meta")
    title = metadata.get("title", "") if isinstance(metadata, dict) else ""
    searchable_values = (node.get("class_type", ""), node.get("title", ""), title)
    searchable_text = " ".join(str(value).lower() for value in searchable_values)
    compact_text = searchable_text.replace("-", "").replace("_", "").replace(" ", "")

    return "qwen3" in compact_text and "voiceclone" in compact_text


def replace_voice_clone_text(workflow: dict[str, Any], text: str) -> dict[str, Any]:
    """Return a workflow copy with the Qwen3-TTS VoiceClone target text replaced."""

    updated_workflow = copy.deepcopy(workflow)

    for node in updated_workflow.values():
        if not isinstance(node, dict) or not node_matches_qwen3_voice_clone(node):
            continue

        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise HTTPException(
                status_code=500,
                detail="Qwen3-TTS VoiceClone node does not contain an inputs object.",
            )

        for field_name in TARGET_TEXT_FIELDS:
            if field_name in inputs:
                inputs[field_name] = text
                return updated_workflow

        raise HTTPException(
            status_code=500,
            detail=(
                "Qwen3-TTS VoiceClone node is missing a target text input "
                f"field. Expected one of: {', '.join(TARGET_TEXT_FIELDS)}."
            ),
        )

    raise HTTPException(
        status_code=500,
        detail="Could not find a Qwen3-TTS VoiceClone node in the workflow.",
    )


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

    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI rejected the prompt with HTTP {exc.code}: {detail}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to ComfyUI at {prompt_url}: {exc.reason}",
        ) from exc

    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ComfyUIError("ComfyUI returned a non-JSON response.") from exc

    prompt_id = response_json.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ComfyUIError("ComfyUI response did not include a prompt_id.")

    return prompt_id


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a simple health check response."""

    return {"status": "ok"}


@app.post("/tts", response_model=TTSResponse, tags=["tts"])
def create_tts(request: TTSRequest) -> TTSResponse:
    """Submit the Qwen3-TTS workflow to ComfyUI and return its prompt id."""

    workflow = load_workflow()
    updated_workflow = replace_voice_clone_text(workflow, request.text)

    try:
        prompt_id = submit_prompt_to_comfyui(updated_workflow)
    except ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TTSResponse(prompt_id=prompt_id)
