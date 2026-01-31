import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(zcr),
        np.mean(spectral_centroid)
    ])

    return features
