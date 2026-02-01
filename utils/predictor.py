import joblib
import numpy as np
import librosa

MODEL_PATH = "models/voice_detector.pkl"
model = joblib.load(MODEL_PATH)


def extract_features(file_path):
    # Load up to first 5 seconds (balanced accuracy + speed)
    y, sr = librosa.load(file_path, sr=None, duration=5.0)

    # Pad very short clips
    if len(y) < sr * 0.5:
        y = np.pad(y, (0, int(sr * 0.5) - len(y)))

    # Nyquist safeguard
    nyquist = sr / 2
    safe_fmax = min(8000, nyquist - 100)

    # Faster FFT settings
    n_fft = 1024
    hop_length = 512

    # MFCC (reduced from 40 → 20 for speed)
    mfcc = np.mean(
        librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=20,
            n_fft=n_fft, hop_length=hop_length,
            fmax=safe_fmax
        ).T,
        axis=0
    )

    # Chroma
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=y, sr=sr,
            n_fft=n_fft, hop_length=hop_length
        ).T,
        axis=0
    )

    # Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T,
        axis=0
    )

    # Spectral Rolloff
    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=y, sr=sr,
            n_fft=n_fft, hop_length=hop_length
        ).T,
        axis=0
    )

    return np.hstack([mfcc, chroma, zcr, rolloff])


def predict_audio(file_path):
    features = extract_features(file_path).reshape(1, -1)

    probs = model.predict_proba(features)[0]
    human_prob = probs[0]
    ai_prob = probs[1]

    # Smarter threshold decision
    if ai_prob > 0.55:
        label = "AI_GENERATED"
        confidence = ai_prob
        explanation = "Spectral smoothness and reduced natural variability indicate AI-generated speech."
    else:
        label = "HUMAN"
        confidence = human_prob
        explanation = "Natural pitch fluctuations and human vocal irregularities detected."

    return label, float(confidence), explanation