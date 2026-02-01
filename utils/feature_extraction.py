import librosa
import numpy as np

def extract_features(file_path):
    # Load audio with original sample rate to avoid mismatch issues
    y, sr = librosa.load(file_path, sr=None)

    # Safety check: ensure audio is long enough
    if len(y) < sr * 0.5:  # less than 0.5 seconds
        y = np.pad(y, (0, int(sr * 0.5) - len(y)))

    # Nyquist frequency safeguard
    nyquist = sr / 2
    safe_fmax = min(8000, nyquist - 100)

    # MFCC features (core voice features)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        fmax=safe_fmax
    )

    # Zero Crossing Rate (speech noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)

    # Spectral Centroid (brightness of voice)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    # Spectral Bandwidth (spread of frequencies)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    # Spectral Rolloff (high frequency energy)
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    # Combine all features into one vector
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(zcr),
        np.mean(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.mean(spectral_rolloff)
    ])

    return features