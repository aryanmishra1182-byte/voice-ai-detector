from fastapi.responses import FileResponse
import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import os

from utils.audio_processing import base64_to_wav
from utils.predictor import predict_audio
from security.auth import verify_api_key

print("🚀 FastAPI app starting...")

app = FastAPI(title="Voice AI Detection API")

# 🌍 CORS (allow frontend + swagger)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 API Key security scheme for Swagger
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# 🌐 Root route
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join("frontend", "index.html"))


SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.post("/api/voice-detection")
def detect_voice(
    req: VoiceRequest,
    api_key: str = Security(api_key_header)
):
    verify_api_key(api_key)

    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    try:
        wav_path = base64_to_wav(req.audioBase64)
        label, confidence, explanation = predict_audio(wav_path)
        os.remove(wav_path)
    except Exception as e:
        print("Processing error:", str(e))
        raise HTTPException(status_code=500, detail="Audio processing failed")

    return {
        "status": "success",
        "language": req.language,
        "classification": label,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }

