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

## Test

Install development dependencies and run the test suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```
