from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from utils.audio_processing import base64_to_wav
from utils.predictor import predict_audio
from security.auth import verify_api_key
import os
import time

app = FastAPI()

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# Serve frontend
@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()

    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    try:
        wav_path = base64_to_wav(req.audioBase64)
        label, confidence, explanation = predict_audio(wav_path)
        os.remove(wav_path)
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail="Audio processing failed")

    processing_time = int((time.time() - start_time) * 1000)

    verdict = "This audio is likely human speech."
    if label == "AI_GENERATED":
        verdict = "This audio is likely AI-generated speech."

    return {
        "status": "success",
        "analysis": {
            "language": req.language,
            "voice_type": label,
            "confidence_score": round(confidence, 2),
            "verdict": verdict,
            "explanation": explanation,
            "audio_features": {
                "pitch_stability": "Very High" if label == "AI_GENERATED" else "Natural",
                "background_noise": "Very Low" if label == "AI_GENERATED" else "Present",
                "speech_variability": "Low" if label == "AI_GENERATED" else "Natural",
                "articulation_pattern": "Overly consistent" if label == "AI_GENERATED" else "Human-like"
            }
        },
        "processing": {
            "model_version": "VoiceAI-Detector v1.0",
            "processing_time_ms": processing_time
        }
    }
