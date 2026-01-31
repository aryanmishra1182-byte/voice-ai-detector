import numpy as np
import joblib
from utils.feature_extraction import extract_features

# Load lightweight sklearn model
model = joblib.load("model/voice_model.pkl")

def predict_audio(file_path):
    features = extract_features(file_path)
    features = np.array(features).reshape(1, -1)

    probs = model.predict_proba(features)[0]
    confidence = float(max(probs))
    label = "AI_GENERATED" if probs[1] > probs[0] else "HUMAN"

    explanation = (
        "Unnaturally stable acoustic patterns detected"
        if label == "AI_GENERATED"
        else "Natural human pitch variations detected"
    )

    return label, confidence, explanation
