import joblib
import numpy as np
import librosa

MODEL_PATH = "models/voice_detector.pkl"
model = joblib.load(MODEL_PATH)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, spectral, zcr, rolloff])

def predict_audio(file_path):
    features = extract_features(file_path).reshape(1, -1)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    confidence = prob[pred]

    if pred == 1:
        label = "AI_GENERATED"
        explanation = "Synthetic speech patterns and spectral smoothness detected"
    else:
        label = "HUMAN"
        explanation = "Natural pitch variation and human vocal irregularities detected"

    return label, float(confidence), explanation
