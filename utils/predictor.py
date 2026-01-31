import torch
import torch.nn.functional as F
import numpy as np
from utils.feature_extraction import extract_features

class VoiceClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

# Load one sample file to detect feature size
sample_feature = extract_features("dataset/human/english/" + 
    next(iter(__import__("os").listdir("dataset/human/english"))))
input_size = len(sample_feature)

model = VoiceClassifier(input_size)
model.load_state_dict(torch.load("model/voice_model.pt", map_location="cpu"))
model.eval()

def predict_audio(file_path):
    features = extract_features(file_path)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = model(features)
        probs = F.softmax(outputs, dim=1).numpy()[0]

    confidence = float(probs[1])
    label = "AI_GENERATED" if probs[1] > probs[0] else "HUMAN"

    explanation = (
        "Unnaturally stable acoustic patterns detected"
        if label == "AI_GENERATED"
        else "Natural human pitch variations detected"
    )

    return label, confidence, explanation
