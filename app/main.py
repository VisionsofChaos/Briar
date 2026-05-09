"""FastAPI service entry point for Briar."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Briar TTS Service",
    version="0.1.0",
    description="A placeholder API surface for future text-to-speech generation.",
)


class TTSRequest(BaseModel):
    """Request body for placeholder text-to-speech generation."""

    text: str = Field(..., min_length=1, description="Text to synthesize.")


class TTSResponse(BaseModel):
    """Placeholder response returned until audio generation is implemented."""

    status: Literal["placeholder"]
    message: str
    text: str
    audio_url: str | None = None


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a simple health check response."""

    return {"status": "ok"}


@app.post("/tts", response_model=TTSResponse, tags=["tts"])
def create_tts(request: TTSRequest) -> TTSResponse:
    """Accept text and return a placeholder TTS response."""

    return TTSResponse(
        status="placeholder",
        message="TTS generation is not implemented yet.",
        text=request.text,
    )
