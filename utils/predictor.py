import joblib
import numpy as np
import librosa

MODEL_PATH = "models/voice_detector.pkl"
model = joblib.load(MODEL_PATH)


def extract_features(file_path):
    try:
        # Load only first 5 seconds for speed (SAFE optimization)
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)

        # Ensure minimum length
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)))

        # === FEATURE SET (MUST MATCH TRAINING) ===
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)  # 40
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)    # 12
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)  # 7
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))  # 1
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))  # 1

        features = np.hstack([mfcc, chroma, contrast, zcr, rolloff])

        return features

    except Exception as e:
        print("Feature extraction error:", e)
        raise


def predict_audio(file_path):
    features = extract_features(file_path).reshape(1, -1)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = float(np.max(prob))

    if pred == 1:
        label = "AI_GENERATED"
        explanation = "Synthetic speech patterns and spectral smoothness detected"
    else:
        label = "HUMAN"
        explanation = "Natural pitch variation and human vocal irregularities detected"

    return label, confidence, explanation