import joblib
import numpy as np
import librosa

MODEL_PATH = "models/voice_detector.pkl"
model = joblib.load(MODEL_PATH)


def extract_features(file_path):
    try:
        # LOAD AUDIO EXACTLY LIKE TRAINING
        y, sr = librosa.load(file_path, sr=None)

        # Limit very long audio for performance (optional safe trim)
        if len(y) > sr * 8:  # use max first 8 seconds
            y = y[:sr * 8]

        # ===== SAME FEATURES USED DURING TRAINING =====

        # 40 MFCC
        mfcc = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
            axis=0
        )

        # 12 Chroma
        chroma = np.mean(
            librosa.feature.chroma_stft(y=y, sr=sr).T,
            axis=0
        )

        # 7 Spectral Contrast
        spectral = np.mean(
            librosa.feature.spectral_contrast(y=y, sr=sr).T,
            axis=0
        )

        # 1 Zero Crossing Rate
        zcr = np.mean(
            librosa.feature.zero_crossing_rate(y).T,
            axis=0
        )

        # 1 Spectral Rolloff
        rolloff = np.mean(
            librosa.feature.spectral_rolloff(y=y, sr=sr).T,
            axis=0
        )

        # TOTAL = 61 FEATURES (MUST MATCH MODEL)
        features = np.hstack([mfcc, chroma, spectral, zcr, rolloff])

        return features

    except Exception as e:
        print("Feature extraction error:", e)
        raise


def predict_audio(file_path):
    try:
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

    except Exception as e:
        print("Prediction error:", e)
        raise