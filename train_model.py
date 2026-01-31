import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.feature_extraction import extract_features

DATASET_PATH = "dataset"

X = []
y = []

for label_type in ["human", "ai"]:
    label = 0 if label_type == "human" else 1
    folder = os.path.join(DATASET_PATH, label_type)

    for language in os.listdir(folder):
        lang_path = os.path.join(folder, language)

        for file in os.listdir(lang_path):
            file_path = os.path.join(lang_path, file)
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
                print("Processed:", file_path)
            except:
                print("Skipped:", file_path)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

class VoiceClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

model = VoiceClassifier(X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/voice_model.pt")
print("Model saved!")
