import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Create dummy training data
X = np.random.rand(300, 42)
y = np.random.randint(0, 2, 300)

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model/voice_model.pkl")
print("Lightweight model saved!")
