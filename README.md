# Briar

Briar includes a small FastAPI service that submits a Qwen3-TTS workflow to a local ComfyUI server and can pair LM Studio chat replies with generated speech.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Workflow

Export your ComfyUI workflow in API format and save it as:

```text
workflows\qwen3_tts.json
```

The workflow must contain a Qwen3-TTS VoiceClone node with a target text input such as `target_text` and a Save Audio node that writes the generated audio. When `POST /tts` is called, Briar loads that workflow, replaces the Qwen3-TTS VoiceClone target text with the incoming request text, submits the workflow to ComfyUI at `http://127.0.0.1:8188/prompt`, and polls `http://127.0.0.1:8188/history/{prompt_id}` until ComfyUI reports a Save Audio output. Briar recursively scans each output node for any nested dictionary containing a non-empty `filename`, including common ComfyUI keys such as `audio`, `audios`, and `files`, so audio files saved in subfolders like `audio` are detected.

## Run ComfyUI

Start ComfyUI separately and keep it listening on the default local address:

```text
http://127.0.0.1:8188
```


## Run LM Studio

For `POST /chat`, start LM Studio's local OpenAI-compatible server and load the `qwen2.5-14b` model. Briar sends chat completions to:

```text
http://127.0.0.1:50123/v1/chat/completions
```

## Run the API

Start the development server with Python's module runner so the command works consistently on Windows, macOS, and Linux:

```powershell
python -m uvicorn app.main:app --reload
```

The service will be available at <http://127.0.0.1:8000>.

## Endpoints

### `GET /health`

Returns a simple health check response:

```json
{"status":"ok"}
```

### `POST /tts`

Accepts text for text-to-speech generation, submits the configured ComfyUI workflow, waits up to 120 seconds for an audio output, and returns the ComfyUI `prompt_id` plus the generated audio location. Briar returns a clear HTTP 504 error if no audio appears before the timeout.

Request:

```json
{"text":"Hello from Briar"}
```

Response:

```json
{
  "prompt_id": "00000000-0000-0000-0000-000000000000",
  "filename": "generated.wav",
  "subfolder": "",
  "type": "output",
  "audio_url": "http://127.0.0.1:8188/view?filename=generated.wav&subfolder=&type=output"
}
```

### `POST /chat`

Accepts a user message, sends it to LM Studio using model `qwen2.5-14b`, synthesizes Briar's reply through the same ComfyUI TTS workflow, and returns both the text reply and generated audio location. LM Studio failures return HTTP 502; TTS generation timeouts return HTTP 504.

Request:

```json
{"message":"How are you?"}
```

Response:

```json
{
  "message": "How are you?",
  "reply": "I'm steady and glad you asked.",
  "prompt_id": "00000000-0000-0000-0000-000000000000",
  "filename": "generated.wav",
  "subfolder": "",
  "type": "output",
  "audio_url": "http://127.0.0.1:8188/view?filename=generated.wav&subfolder=&type=output"
}
```

## Test

Install development dependencies and run the test suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```
