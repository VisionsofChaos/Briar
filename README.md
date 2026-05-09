# Briar

Briar includes a small FastAPI service that submits a Qwen3-TTS workflow to a local ComfyUI server.

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

The workflow must contain a Qwen3-TTS node with a target text input such as `target_text`. When `POST /tts` is called, Briar loads that workflow, replaces the Qwen3-TTS target text with the incoming request text, and submits the workflow to ComfyUI at `http://127.0.0.1:8188/prompt`.

## Run ComfyUI

Start ComfyUI separately and keep it listening on the default local address:

```text
http://127.0.0.1:8188
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

Accepts text for text-to-speech generation, submits the configured ComfyUI workflow, and returns the ComfyUI `prompt_id`. Briar does not poll ComfyUI history or fetch generated audio yet.

Request:

```json
{"text":"Hello from Briar"}
```

Response:

```json
{
  "prompt_id": "00000000-0000-0000-0000-000000000000"
}
```

## Test

Install development dependencies and run the test suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```
