"""FastAPI service entry point for Briar."""

import copy
import json
import logging
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
    "You are Briar. You are calm, emotionally intelligent, witty, warm, and "
    "grounded. You speak naturally and conversationally. Do not mention being an "
    "AI, a model, an assistant, or roleplaying. Do not output reasoning or "
    "chain-of-thought. Reply naturally."
)
DEFAULT_HISTORY_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 1
WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "workflows" / "qwen3_tts.json"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
TARGET_TEXT_FIELDS = ("target_text", "text", "prompt")
logger = logging.getLogger(__name__)

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
      background: #160f12;
      color: #fff7ed;
      --panel: rgba(35, 22, 25, 0.68);
      --panel-strong: rgba(28, 18, 21, 0.82);
      --cream: #fff0d6;
      --rose: #d9a0a8;
      --rose-bright: #efc0c8;
      --wine: #3a2028;
      --ink: #1a1013;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      overflow: hidden;
      background-color: #170f13;
      background-image:
        linear-gradient(90deg, rgba(18, 9, 12, 0.48), rgba(18, 9, 12, 0.14) 48%, rgba(18, 9, 12, 0.72)),
        radial-gradient(circle at 26% 28%, rgba(255, 220, 185, 0.20), transparent 24rem),
        radial-gradient(circle at 80% 18%, rgba(202, 126, 139, 0.18), transparent 28rem),
        url('/static/backgrounds/briar_room.png'),
        linear-gradient(135deg, #1a1014 0%, #2b1820 48%, #120c10 100%);
      background-size: cover, auto, auto, cover, cover;
      background-position: center, center, center, center, center;
      background-repeat: no-repeat;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(180deg, rgba(255, 240, 214, 0.06), rgba(0, 0, 0, 0.20));
      backdrop-filter: saturate(1.04);
    }
    main {
      position: relative;
      z-index: 1;
      width: min(1180px, calc(100vw - 2rem));
      height: min(820px, calc(100vh - 2rem));
      display: grid;
      grid-template-rows: auto 1fr auto auto auto;
      gap: 0.85rem;
      padding: clamp(1rem, 2vw, 1.45rem);
      border: 1px solid rgba(255, 240, 214, 0.22);
      border-radius: 30px;
      background: linear-gradient(135deg, rgba(45, 28, 32, 0.50), rgba(18, 12, 15, 0.42));
      box-shadow: 0 28px 90px rgba(0, 0, 0, 0.50), inset 0 1px 0 rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(10px);
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 1rem;
      padding: 0 0.25rem;
    }
    header h1 {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2.2rem, 5vw, 4.2rem);
      line-height: 0.9;
      letter-spacing: 0.03em;
      color: var(--cream);
      text-shadow: 0 4px 22px rgba(0, 0, 0, 0.55);
    }
    header p {
      margin: 0.35rem 0 0;
      color: rgba(255, 240, 214, 0.76);
      max-width: 34rem;
    }
    .stage {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(340px, 46%) minmax(340px, 1fr);
      gap: clamp(1rem, 2.6vw, 2rem);
      align-items: stretch;
    }
    .portrait-card {
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: end;
      min-height: 0;
      padding: clamp(0.8rem, 1.6vw, 1.2rem);
      border-radius: 28px;
      background:
        radial-gradient(circle at 50% 22%, rgba(255, 229, 205, 0.18), transparent 20rem),
        linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.015));
      border: 1px solid rgba(255, 240, 214, 0.16);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
      overflow: hidden;
    }
    .portrait-card::after {
      content: "";
      position: absolute;
      left: 10%;
      right: 10%;
      bottom: 1rem;
      height: 20%;
      border-radius: 50%;
      background: radial-gradient(ellipse, rgba(0, 0, 0, 0.34), transparent 68%);
      filter: blur(10px);
      z-index: 0;
    }
    .portrait-frame {
      position: relative;
      z-index: 1;
      width: min(540px, 112%);
      height: 100%;
      min-height: 500px;
      display: grid;
      place-items: end center;
      overflow: visible;
      border-radius: 0;
      background: transparent;
      border: 0;
    }
    #briar-canvas,
    #briar-portrait {
      width: min(540px, 112%);
      height: 100%;
      object-fit: contain;
      object-position: bottom center;
      filter: drop-shadow(0 28px 34px rgba(0, 0, 0, 0.54));
    }
    #briar-canvas { display: block; }
    #briar-portrait { display: none; }
    .portrait-fallback {
      display: none;
      width: min(300px, 80%);
      aspect-ratio: 1 / 1;
      place-items: center;
      border-radius: 999px;
      background: rgba(42, 25, 30, 0.74);
      border: 1px solid rgba(255, 240, 214, 0.16);
      font-size: clamp(4rem, 16vw, 8rem);
      font-family: Georgia, "Times New Roman", serif;
      font-weight: 900;
      color: var(--cream);
    }
    .portrait-frame.fallback-image #briar-canvas { display: none; }
    .portrait-frame.fallback-image #briar-portrait { display: block; }
    .portrait-frame.missing #briar-canvas,
    .portrait-frame.missing #briar-portrait { display: none; }
    .portrait-frame.missing .portrait-fallback { display: grid; }
    .portrait-caption {
      position: relative;
      z-index: 1;
      margin: 0.25rem 0 0;
      padding: 0.45rem 0.85rem;
      border-radius: 999px;
      color: rgba(255, 240, 214, 0.78);
      background: rgba(26, 16, 19, 0.42);
      border: 1px solid rgba(255, 240, 214, 0.12);
      text-align: center;
      font-size: 0.95rem;
    }
    #chat-log {
      min-height: 0;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      padding: clamp(1rem, 1.8vw, 1.25rem);
      border-radius: 28px;
      background: linear-gradient(180deg, rgba(36, 23, 27, 0.78), rgba(20, 13, 16, 0.70));
      border: 1px solid rgba(255, 240, 214, 0.18);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 18px 48px rgba(0, 0, 0, 0.22);
      backdrop-filter: blur(12px);
    }
    #chat-log:empty::before {
      content: "The room is quiet. Say hello to Briar.";
      margin: auto;
      color: rgba(255, 240, 214, 0.50);
      text-align: center;
      font-style: italic;
    }
    .message {
      max-width: 82%;
      padding: 0.9rem 1rem;
      border-radius: 18px;
      line-height: 1.48;
      white-space: pre-wrap;
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.18);
    }
    .message.user {
      align-self: flex-end;
      background: linear-gradient(135deg, #f4ddc5, #e7c0ba);
      color: #2a171d;
      border-bottom-right-radius: 5px;
    }
    .message.briar {
      align-self: flex-start;
      background: rgba(255, 240, 214, 0.10);
      border: 1px solid rgba(255, 240, 214, 0.16);
      color: #fff8ed;
      border-bottom-left-radius: 5px;
    }
    .meta {
      display: block;
      margin-bottom: 0.2rem;
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      opacity: 0.68;
      font-weight: 800;
    }
    form {
      display: flex;
      gap: 0.75rem;
      padding: 0.35rem;
      border-radius: 999px;
      background: rgba(24, 15, 18, 0.66);
      border: 1px solid rgba(255, 240, 214, 0.15);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    input {
      flex: 1;
      min-width: 0;
      padding: 0.9rem 1rem;
      border: 0;
      border-radius: 999px;
      background: transparent;
      color: inherit;
      font: inherit;
      outline: none;
    }
    input::placeholder { color: rgba(255, 240, 214, 0.46); }
    input:focus { box-shadow: 0 0 0 3px rgba(217, 160, 168, 0.18); }
    button {
      padding: 0.9rem 1.35rem;
      border: 0;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--rose-bright), #f0d4be);
      color: #2a171d;
      font: inherit;
      font-weight: 900;
      cursor: pointer;
      box-shadow: 0 10px 28px rgba(91, 43, 52, 0.30);
    }
    button:disabled { cursor: wait; opacity: 0.62; }
    #status {
      min-height: 1.35rem;
      padding: 0 0.35rem;
      color: rgba(255, 240, 214, 0.68);
    }
    #status.error { color: #ffb4a8; }
    audio {
      width: 100%;
      height: 38px;
      opacity: 0.84;
      filter: sepia(0.18) saturate(0.86);
    }
    @media (max-width: 860px) {
      body { overflow: auto; place-items: stretch; }
      main {
        min-height: 100vh;
        height: auto;
        width: 100vw;
        border-radius: 0;
        padding: 0.85rem;
      }
      header { display: block; }
      .stage { grid-template-columns: 1fr; grid-template-rows: auto minmax(320px, 1fr); }
      .portrait-card { min-height: 340px; padding: 0.6rem; }
      .portrait-frame { min-height: 300px; width: min(360px, 100%); }
      #briar-canvas, #briar-portrait { width: min(360px, 108%); }
      .portrait-caption { font-size: 0.86rem; }
      .message { max-width: 92%; }
      form { border-radius: 24px; align-items: stretch; }
      button { padding-inline: 1rem; }
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
          <canvas id="briar-canvas" aria-label="Briar portrait"></canvas>
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
    const canvas = document.getElementById('briar-canvas');
    const canvasContext = canvas.getContext ? canvas.getContext('2d', { willReadFrequently: true }) : null;
    const portraitFrame = document.getElementById('portrait-frame');
    const visemeBasePath = '/static/visemes';
    const spriteMap = {
      idle: { normal: 'briar_idle.png', blink: 'briar_idle_blink.png' },
      e: { normal: 'briar_e.png', blink: 'briar_e_blink.png' },
      a: { normal: 'briar_a.png', blink: 'briar_a_blink.png' },
      open: { normal: 'briar_open.png', blink: 'briar_open_blink.png' },
      o: { normal: 'briar_o.png', blink: 'briar_o_blink.png' },
    };
    const weightedMouths = ['a', 'a', 'a', 'e', 'e', 'e', 'open', 'o'];
    const failedSprites = new Set();
    const spriteImages = {};
    let currentMouth = 'idle';
    let isBlinking = false;
    let lipSyncTimer = null;
    let blinkTimer = null;
    let blinkEndTimer = null;
    let currentSpriteSrc = '';

    function spriteUrl(filename) {
      return `${visemeBasePath}/${filename}`;
    }

    function selectSpriteFilename() {
      const sprites = spriteMap[currentMouth] || spriteMap.idle;
      if (isBlinking && !failedSprites.has(sprites.blink)) {
        return sprites.blink;
      }
      return sprites.normal;
    }

    function showImageFallback(filename) {
      currentSpriteSrc = spriteUrl(filename);
      portraitFrame.classList.remove('missing');
      portraitFrame.classList.add('fallback-image');
      portrait.src = currentSpriteSrc;
    }

    function chromakeyImageData(imageData) {
      const pixels = imageData.data;
      for (let index = 0; index < pixels.length; index += 4) {
        const r = pixels[index];
        const g = pixels[index + 1];
        const b = pixels[index + 2];
        if (g > 120 && g > r * 1.25 && g > b * 1.25) {
          pixels[index + 3] = 0;
        } else if (g > 105 && g > r * 1.12 && g > b * 1.12) {
          pixels[index + 3] = Math.min(pixels[index + 3], 150);
        }
      }
      return imageData;
    }

    function renderBriarSprite(filename = selectSpriteFilename()) {
      const image = spriteImages[filename];
      if (!canvasContext || !image) {
        showImageFallback(filename);
        return;
      }
      if (!image.complete || image.naturalWidth === 0) {
        return;
      }

      try {
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        canvasContext.clearRect(0, 0, canvas.width, canvas.height);
        canvasContext.drawImage(image, 0, 0);
        const imageData = canvasContext.getImageData(0, 0, canvas.width, canvas.height);
        canvasContext.putImageData(chromakeyImageData(imageData), 0, 0);
        currentSpriteSrc = spriteUrl(filename);
        portrait.src = currentSpriteSrc;
        portraitFrame.classList.remove('missing', 'fallback-image');
      } catch (error) {
        showImageFallback(filename);
      }
    }

    function preloadSprites() {
      Object.values(spriteMap).forEach((sprites) => {
        Object.values(sprites).forEach((filename) => {
          const image = new Image();
          spriteImages[filename] = image;
          image.addEventListener('load', () => {
            if (filename === selectSpriteFilename()) {
              renderBriarSprite(filename);
            }
          });
          image.addEventListener('error', () => {
            failedSprites.add(filename);
            if (filename.includes('_blink')) {
              updateBriarSprite();
              return;
            }
            if (filename === selectSpriteFilename()) {
              portraitFrame.classList.add('missing');
            }
          });
          image.src = spriteUrl(filename);
        });
      });
    }

    function updateBriarSprite() {
      const filename = selectSpriteFilename();
      const nextSrc = spriteUrl(filename);
      if (currentSpriteSrc !== nextSrc || portraitFrame.classList.contains('fallback-image')) {
        renderBriarSprite(filename);
      }
    }

    function chooseSpeakingMouth() {
      return weightedMouths[Math.floor(Math.random() * weightedMouths.length)];
    }

    function stopLipSync() {
      if (lipSyncTimer) {
        clearTimeout(lipSyncTimer);
        lipSyncTimer = null;
      }
      currentMouth = 'idle';
      updateBriarSprite();
    }

    function scheduleLipSyncFrame() {
      currentMouth = chooseSpeakingMouth();
      updateBriarSprite();
      const nextDelayMs = 90 + Math.floor(Math.random() * 41);
      lipSyncTimer = setTimeout(scheduleLipSyncFrame, nextDelayMs);
    }

    function startLipSync() {
      if (lipSyncTimer) return;
      scheduleLipSyncFrame();
    }

    function scheduleNextBlink() {
      const nextBlinkDelayMs = 3000 + Math.floor(Math.random() * 4001);
      blinkTimer = setTimeout(() => {
        isBlinking = true;
        updateBriarSprite();
        const blinkDurationMs = 90 + Math.floor(Math.random() * 41);
        blinkEndTimer = setTimeout(() => {
          isBlinking = false;
          updateBriarSprite();
          scheduleNextBlink();
        }, blinkDurationMs);
      }, nextBlinkDelayMs);
    }

    portrait.addEventListener('error', () => {
      const failedFilename = currentSpriteSrc.split('/').pop();
      failedSprites.add(failedFilename);
      if (failedFilename && failedFilename.includes('_blink')) {
        updateBriarSprite();
        return;
      }
      portraitFrame.classList.add('missing');
    });
    preloadSprites();
    updateBriarSprite();
    scheduleNextBlink();
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
    description="A small API surface for submitting text-to-speech jobs to ComfyUI.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


class ChatRequest(BaseModel):
    """Request body for chat plus text-to-speech generation."""

    message: str = Field(..., min_length=1, description="Message to send to Briar.")


class ChatResponse(TTSResponse):
    """Response returned after chat completion and TTS generation."""

    message: str
    reply: str


class AudioOutput(BaseModel):
    """Generated audio location returned by ComfyUI history."""

    filename: str
    subfolder: str = ""
    type: str = "output"


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI does not return the expected response."""


class LMStudioError(RuntimeError):
    """Raised when LM Studio does not return the expected response."""


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


def replace_qwen3_target_text(workflow: dict[str, Any], text: str) -> dict[str, Any]:
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
                "Qwen3-TTS VoiceClone node is missing a target text input field. "
                f"Expected one of: {', '.join(TARGET_TEXT_FIELDS)}."
            ),
        )

    raise HTTPException(
        status_code=500,
        detail="Could not find a Qwen3-TTS VoiceClone node in the workflow.",
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


def request_chat_completion(
    message: str, chat_url: str = LM_STUDIO_CHAT_URL
) -> str:
    """Send a user message to LM Studio and return the assistant reply text."""

    request_body = json.dumps(
        {
            "model": LM_STUDIO_MODEL,
            "messages": [
                {"role": "system", "content": BRIAR_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "temperature": 0.8,
        }
    ).encode("utf-8")
    request = Request(
        chat_url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"LM Studio returned HTTP {exc.code}: {detail}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to LM Studio at {chat_url}: {exc.reason}",
        ) from exc

    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise LMStudioError("LM Studio returned a non-JSON response.") from exc

    if not isinstance(response_json, dict):
        raise LMStudioError("LM Studio returned JSON that was not an object.")

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
    """Build an audio output model from one ComfyUI Save Audio item."""

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
    """Recursively return dicts containing filenames from ComfyUI history output."""

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


def history_debug_details(history: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    """Return ComfyUI history structure details for timeout debugging."""

    prompt_history = history.get(prompt_id)
    outputs = prompt_history.get("outputs") if isinstance(prompt_history, dict) else None

    output_keys_by_node: dict[str, list[str]] = {}
    if isinstance(outputs, dict):
        output_keys_by_node = {
            str(node_id): list(node_output.keys())
            for node_id, node_output in outputs.items()
            if isinstance(node_output, dict)
        }

    return {
        "prompt_id": prompt_id,
        "history_keys": list(history.keys()),
        "output_node_ids": list(outputs.keys()) if isinstance(outputs, dict) else [],
        "output_keys_by_node": output_keys_by_node,
    }


def extract_audio_output(history: dict[str, Any], prompt_id: str) -> AudioOutput | None:
    """Extract the first generated audio/file output from ComfyUI history."""

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
    """Poll ComfyUI history until an audio output appears or timeout expires."""

    history_url = f"{base_url}/history/{prompt_id}"
    deadline = time.monotonic() + timeout_seconds
    last_history: dict[str, Any] = {}

    while True:
        request = Request(history_url, method="GET")
        history = read_json_from_comfyui(request)
        last_history = history
        audio_output = extract_audio_output(history, prompt_id)
        if audio_output is not None:
            return audio_output

        if time.monotonic() >= deadline:
            debug_details = history_debug_details(last_history, prompt_id)
            logger.warning(
                "Timed out waiting for ComfyUI audio output. "
                "prompt_id=%s history_keys=%s output_node_ids=%s output_keys_by_node=%s",
                debug_details["prompt_id"],
                debug_details["history_keys"],
                debug_details["output_node_ids"],
                debug_details["output_keys_by_node"],
            )
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


@app.get("/", response_class=HTMLResponse, tags=["frontend"])
def home() -> HTMLResponse:
    """Serve the local Briar chat frontend."""

    return HTMLResponse(FRONTEND_HTML)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a simple health check response."""

    return {"status": "ok"}


def synthesize_text(text: str) -> TTSResponse:
    """Run the configured ComfyUI TTS workflow for text and return audio details."""

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


@app.post("/tts", response_model=TTSResponse, tags=["tts"])
def create_tts(request: TTSRequest) -> TTSResponse:
    """Submit the Qwen3-TTS workflow to ComfyUI and return generated audio details."""

    try:
        return synthesize_text(request.text)
    except ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def create_chat(request: ChatRequest) -> ChatResponse:
    """Generate a Briar chat reply and synthesize it with the TTS workflow."""

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
