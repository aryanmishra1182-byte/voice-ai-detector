import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATASET_PATH = "dataset"
MODEL_PATH = "models/voice_detector.pkl"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)

        return np.hstack([mfcc, chroma, spectral, zcr, rolloff])
    except Exception as e:
        print("Error processing:", file_path, e)
        return None

features = []
labels = []

print("📂 Loading dataset...")

for label_name in ["human", "ai"]:
    label_folder = os.path.join(DATASET_PATH, label_name)
    label_value = 0 if label_name == "human" else 1

    for root, dirs, files in os.walk(label_folder):
        for file in files:
            if file.endswith(".mp3"):
                path = os.path.join(root, file)
                feats = extract_features(path)
                if feats is not None:
                    features.append(feats)
                    labels.append(label_value)

X = np.array(features)
y = np.array(labels)

print("Total samples:", len(X))
print("Human samples:", np.sum(y==0))
print("AI samples:", np.sum(y==1))

if len(set(y)) < 2:
    print("❌ ERROR: Need both HUMAN and AI samples")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, preds))
print("\n📊 Classification Report:\n", classification_report(y_test, preds))

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
