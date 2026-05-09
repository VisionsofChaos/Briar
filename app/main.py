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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


COMFYUI_BASE_URL = "http://127.0.0.1:8188"
COMFYUI_PROMPT_URL = f"{COMFYUI_BASE_URL}/prompt"

LM_STUDIO_CHAT_URL = "http://127.0.0.1:50123/v1/chat/completions"
LM_STUDIO_MODEL = "qwen2.5-14b"

BRIAR_SYSTEM_PROMPT = (
    "You are Briar. You are calm, emotionally intelligent, witty, warm, and grounded. "
    "You speak naturally and conversationally. Do not mention being an AI, a model, "
    "an assistant, or roleplaying. Do not output reasoning or chain-of-thought. "
    "Reply naturally. Keep replies to 1-2 short sentences unless the user asks for depth."
)

DEFAULT_HISTORY_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 1
TTS_MAX_NEW_TOKENS = 512

ROOT_DIR = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT_DIR / "workflows" / "qwen3_tts.json"
STATIC_DIR = ROOT_DIR / "static"
TARGET_TEXT_FIELDS = ("target_text", "text", "prompt")


FRONTEND_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Briar</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #11150f;
      color: #f3f5ed;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top left, rgba(129, 180, 99, 0.22), transparent 32rem),
        linear-gradient(135deg, #10150f 0%, #182016 100%);
    }
    main {
      width: min(1080px, calc(100vw - 2rem));
      height: min(800px, calc(100vh - 2rem));
      display: flex;
      flex-direction: column;
      gap: 1rem;
      padding: 1.25rem;
      border: 1px solid rgba(222, 232, 202, 0.14);
      border-radius: 24px;
      background: rgba(16, 21, 15, 0.82);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
      backdrop-filter: blur(14px);
    }
    header h1 { margin: 0; font-size: clamp(1.8rem, 4vw, 3rem); }
    header p { margin: 0.35rem 0 0; color: #c3cfb3; }
    .stage {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(220px, 320px) 1fr;
      gap: 1rem;
      align-items: stretch;
    }
    .portrait-card {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.8rem;
      padding: 1rem;
      border-radius: 18px;
      background: rgba(0, 0, 0, 0.22);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .portrait-frame {
      width: min(280px, 100%);
      aspect-ratio: 1 / 1;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 28px;
      background: radial-gradient(circle at 50% 18%, rgba(184, 230, 142, 0.22), rgba(0, 0, 0, 0.22));
      border: 1px solid rgba(222, 232, 202, 0.12);
    }
    #briar-portrait {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .portrait-fallback {
      display: none;
      width: 100%;
      height: 100%;
      place-items: center;
      font-size: clamp(4rem, 16vw, 7rem);
      font-weight: 900;
      color: #dcefc8;
    }
    .portrait-frame.missing #briar-portrait { display: none; }
    .portrait-frame.missing .portrait-fallback { display: grid; }
    .portrait-caption { margin: 0; color: #c3cfb3; text-align: center; }
    #chat-log {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      padding: 1rem;
      border-radius: 18px;
      background: rgba(0, 0, 0, 0.22);
    }
    .message {
      max-width: 78%;
      padding: 0.85rem 1rem;
      border-radius: 16px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .message.user {
      align-self: flex-end;
      background: #dcefc8;
      color: #172012;
      border-bottom-right-radius: 4px;
    }
    .message.briar {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-bottom-left-radius: 4px;
    }
    .meta {
      display: block;
      margin-bottom: 0.2rem;
      font-size: 0.8rem;
      opacity: 0.72;
      font-weight: 700;
    }
    form { display: flex; gap: 0.75rem; }
    input {
      flex: 1;
      min-width: 0;
      padding: 0.9rem 1rem;
      border: 1px solid rgba(222, 232, 202, 0.2);
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.26);
      color: inherit;
      font: inherit;
      outline: none;
    }
    input:focus {
      border-color: #b8e68e;
      box-shadow: 0 0 0 3px rgba(184, 230, 142, 0.16);
    }
    button {
      padding: 0.9rem 1.25rem;
      border: 0;
      border-radius: 999px;
      background: #b8e68e;
      color: #172012;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }
    button:disabled { cursor: wait; opacity: 0.6; }
    #status { min-height: 1.35rem; color: #c3cfb3; }
    #status.error { color: #ffb4a8; }
    audio { width: 100%; }
    @media (max-width: 760px) {
      main { height: calc(100vh - 1rem); width: calc(100vw - 1rem); padding: 0.8rem; }
      .stage { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
      .portrait-card { flex-direction: row; justify-content: flex-start; }
      .portrait-frame { width: 120px; border-radius: 20px; }
      .portrait-caption { text-align: left; }
      .message { max-width: 92%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Briar</h1>
      <p>A local voice chat for warm, grounded replies.</p>
    </header>

    <div class="stage">
      <aside class="portrait-card" aria-label="Briar portrait">
        <div id="portrait-frame" class="portrait-frame">
          <img id="briar-portrait" src="/static/visemes/briar_idle.png" alt="Briar portrait">
          <div class="portrait-fallback" aria-hidden="true">B</div>
        </div>
        <p class="portrait-caption">Briar is listening.</p>
      </aside>

      <section id="chat-log" aria-live="polite" aria-label="Chat log"></section>
    </div>

    <div id="status" role="status"></div>
    <audio id="player" controls></audio>

    <form id="chat-form">
      <input id="message-input" name="message" autocomplete="off" placeholder="Say something to Briar..." required>
      <button id="send-button" type="submit">Send</button>
    </form>
  </main>

  <script>
    const form = document.getElementById('chat-form');
    const input = document.getElementById('message-input');
    const button = document.getElementById('send-button');
    const log = document.getElementById('chat-log');
    const status = document.getElementById('status');
    const player = document.getElementById('player');
    const portrait = document.getElementById('briar-portrait');
    const portraitFrame = document.getElementById('portrait-frame');

    const visemeBasePath = '/static/visemes';
    const idleViseme = `${visemeBasePath}/briar_idle.png`;

    const speakingVisemes = [
      'briar_a.png',
      'briar_a.png',
      'briar_a.png',
      'briar_e.png',
      'briar_e.png',
      'briar_e.png',
      'briar_open.png',
      'briar_o.png',
    ];

    let lipSyncTimer = null;

    function setPortrait(src) {
      portraitFrame.classList.remove('missing');
      portrait.src = src;
    }

    function chooseSpeakingViseme() {
      const filename = speakingVisemes[Math.floor(Math.random() * speakingVisemes.length)];
      return `${visemeBasePath}/${filename}`;
    }

    function stopLipSync() {
      if (lipSyncTimer) {
        clearTimeout(lipSyncTimer);
        lipSyncTimer = null;
      }
      setPortrait(idleViseme);
    }

    function scheduleLipSyncFrame() {
      setPortrait(chooseSpeakingViseme());
      const nextDelayMs = 90 + Math.floor(Math.random() * 41);
      lipSyncTimer = setTimeout(scheduleLipSyncFrame, nextDelayMs);
    }

    function startLipSync() {
      if (lipSyncTimer) return;
      scheduleLipSyncFrame();
    }

    portrait.addEventListener('error', () => {
      portraitFrame.classList.add('missing');
    });

    player.addEventListener('play', startLipSync);
    player.addEventListener('pause', stopLipSync);
    player.addEventListener('ended', stopLipSync);
    player.addEventListener('error', stopLipSync);

    function addMessage(role, text) {
      const message = document.createElement('article');
      message.className = `message ${role}`;

      const label = document.createElement('span');
      label.className = 'meta';
      label.textContent = role === 'user' ? 'You' : 'Briar';

      const body = document.createElement('span');
      body.textContent = text;

      message.append(label, body);
      log.appendChild(message);
      log.scrollTop = log.scrollHeight;
    }

    function setStatus(text, isError = false) {
      status.textContent = text;
      status.classList.toggle('error', isError);
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();

      const message = input.value.trim();
      if (!message) return;

      addMessage('user', message);
      input.value = '';
      input.disabled = true;
      button.disabled = true;
      setStatus('Briar is thinking and finding her voice...');

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message }),
        });

        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
          throw new Error(payload.detail || `Request failed with HTTP ${response.status}`);
        }

        addMessage('briar', payload.reply || '');

        if (payload.audio_url) {
          stopLipSync();
          player.src = payload.audio_url;
          await player.play().catch(() => undefined);
        }

        setStatus('');
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error), true);
      } finally {
        input.disabled = false;
        button.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>
"""


app = FastAPI(
    title="Briar TTS Service",
    version="0.1.0",
    description="A small API surface for submitting chat and text-to-speech jobs.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")


class TTSResponse(BaseModel):
    prompt_id: str
    filename: str
    subfolder: str
    type: str
    audio_url: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Message to send to Briar.")


class ChatResponse(TTSResponse):
    message: str
    reply: str


class AudioOutput(BaseModel):
    filename: str
    subfolder: str = ""
    type: str = "output"


class ComfyUIError(RuntimeError):
    pass


class LMStudioError(RuntimeError):
    pass


def load_workflow(workflow_path: Path | None = None) -> dict[str, Any]:
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
    metadata = node.get("_meta")
    title = metadata.get("title", "") if isinstance(metadata, dict) else ""
    searchable_values = (node.get("class_type", ""), node.get("title", ""), title)
    searchable_text = " ".join(str(value).lower() for value in searchable_values)
    compact_text = searchable_text.replace("-", "").replace("_", "").replace(" ", "")
    return "qwen3" in compact_text and "voiceclone" in compact_text


def replace_qwen3_target_text(workflow: dict[str, Any], text: str) -> dict[str, Any]:
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

        did_replace_text = False
        for field_name in TARGET_TEXT_FIELDS:
            if field_name in inputs:
                inputs[field_name] = text
                did_replace_text = True
                break

        if not did_replace_text:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Qwen3-TTS VoiceClone node is missing a target text input field. "
                    f"Expected one of: {', '.join(TARGET_TEXT_FIELDS)}."
                ),
            )

        if "max_new_tokens" in inputs:
            inputs["max_new_tokens"] = TTS_MAX_NEW_TOKENS

        return updated_workflow

    raise HTTPException(
        status_code=500,
        detail="Could not find a Qwen3-TTS VoiceClone node in the workflow.",
    )


def read_json_request(request: Request, service_name: str, timeout: int = 30) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"{service_name} returned HTTP {exc.code}: {detail}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to {service_name} at {request.full_url}: {exc.reason}",
        ) from exc

    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{service_name} returned a non-JSON response.") from exc

    if not isinstance(response_json, dict):
        raise RuntimeError(f"{service_name} returned JSON that was not an object.")

    return response_json


def read_json_from_comfyui(request: Request, timeout: int = 30) -> dict[str, Any]:
    try:
        return read_json_request(request, "ComfyUI", timeout)
    except RuntimeError as exc:
        raise ComfyUIError(str(exc)) from exc


def submit_prompt_to_comfyui(
    workflow: dict[str, Any],
    prompt_url: str = COMFYUI_PROMPT_URL,
) -> str:
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


def request_chat_completion(message: str, chat_url: str = LM_STUDIO_CHAT_URL) -> str:
    request_body = json.dumps(
        {
            "model": LM_STUDIO_MODEL,
            "messages": [
                {"role": "system", "content": BRIAR_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "temperature": 0.8,
            "max_tokens": 80,
        }
    ).encode("utf-8")

    request = Request(
        chat_url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        response_json = read_json_request(request, "LM Studio", timeout=60)
    except RuntimeError as exc:
        raise LMStudioError(str(exc)) from exc

    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LMStudioError("LM Studio response did not include choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LMStudioError("LM Studio response choice was not an object.")

    assistant_message = first_choice.get("message")
    if not isinstance(assistant_message, dict):
        raise LMStudioError("LM Studio response choice did not include a message.")

    reply = assistant_message.get("content")
    if not isinstance(reply, str) or not reply.strip():
        raise LMStudioError("LM Studio response did not include assistant text.")

    return reply.strip()


def audio_output_from_item(audio_item: dict[str, Any]) -> AudioOutput | None:
    filename = audio_item.get("filename")
    if not isinstance(filename, str) or not filename:
        return None

    subfolder = audio_item.get("subfolder", "")
    audio_type = audio_item.get("type", "output")

    return AudioOutput(
        filename=filename,
        subfolder=subfolder if isinstance(subfolder, str) else "",
        type=audio_type if isinstance(audio_type, str) and audio_type else "output",
    )


def iter_save_audio_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        items = [value] if isinstance(value.get("filename"), str) and value["filename"] else []
        for nested_value in value.values():
            items.extend(iter_save_audio_items(nested_value))
        return items

    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for nested_value in value:
            items.extend(iter_save_audio_items(nested_value))
        return items

    return []


def extract_audio_output(history: dict[str, Any], prompt_id: str) -> AudioOutput | None:
    prompt_history = history.get(prompt_id)
    if not isinstance(prompt_history, dict):
        return None

    outputs = prompt_history.get("outputs")
    if not isinstance(outputs, dict):
        return None

    for node_output in outputs.values():
        for audio_item in iter_save_audio_items(node_output):
            audio_output = audio_output_from_item(audio_item)
            if audio_output is not None:
                return audio_output

    return None


def wait_for_audio_output(
    prompt_id: str,
    timeout_seconds: int = DEFAULT_HISTORY_TIMEOUT_SECONDS,
    poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
    base_url: str = COMFYUI_BASE_URL,
) -> AudioOutput:
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
    query = urlencode(
        {
            "filename": audio_output.filename,
            "subfolder": audio_output.subfolder,
            "type": audio_output.type,
        }
    )
    return f"{base_url}/view?{query}"


def synthesize_text(text: str) -> TTSResponse:
    workflow = load_workflow()
    updated_workflow = replace_qwen3_target_text(workflow, text)

    prompt_id = submit_prompt_to_comfyui(updated_workflow)
    audio_output = wait_for_audio_output(prompt_id)

    return TTSResponse(
        prompt_id=prompt_id,
        filename=audio_output.filename,
        subfolder=audio_output.subfolder,
        type=audio_output.type,
        audio_url=build_audio_url(audio_output),
    )


@app.get("/", response_class=HTMLResponse, tags=["frontend"])
def home() -> HTMLResponse:
    return HTMLResponse(FRONTEND_HTML)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tts", response_model=TTSResponse, tags=["tts"])
def create_tts(request: TTSRequest) -> TTSResponse:
    try:
        return synthesize_text(request.text)
    except ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def create_chat(request: ChatRequest) -> ChatResponse:
    try:
        reply = request_chat_completion(request.message)
        tts_response = synthesize_text(reply)
    except LMStudioError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        message=request.message,
        reply=reply,
        prompt_id=tts_response.prompt_id,
        filename=tts_response.filename,
        subfolder=tts_response.subfolder,
        type=tts_response.type,
        audio_url=tts_response.audio_url,
    )