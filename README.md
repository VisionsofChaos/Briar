# Briar

Briar now includes a small FastAPI service with a placeholder text-to-speech endpoint.

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

Accepts text for future text-to-speech generation and currently returns a placeholder JSON response.

Request:

```json
{"text":"Hello from Briar"}
```

Response:

```json
{
  "status": "placeholder",
  "message": "TTS generation is not implemented yet.",
  "text": "Hello from Briar",
  "audio_url": null
}
```

## Test

Install development dependencies and run the test suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```
