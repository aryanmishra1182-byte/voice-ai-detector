from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from utils.audio_processing import base64_to_wav
from utils.predictor import predict_audio
from security.auth import verify_api_key
import os

app = FastAPI()

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, api_key: str = Depends(verify_api_key)):

    # Validate language
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # Validate format
    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    try:
        wav_path = base64_to_wav(req.audioBase64)
        label, confidence, explanation = predict_audio(wav_path)

        # Clean up temp file
        os.remove(wav_path)

    except Exception as e:
        print("Error during processing:", str(e))
        raise HTTPException(status_code=500, detail="Audio processing failed")

    return {
        "status": "success",
        "language": req.language,
        "classification": label,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
