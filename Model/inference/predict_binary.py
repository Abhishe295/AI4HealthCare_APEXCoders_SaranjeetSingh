import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "binary_model.pth")
TEST_DIR = os.path.join(BASE_DIR, "data", "processed", "test")

# ---------------- MODEL ----------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        feats = self.features(x)                # (num_slices, 64, 1, 1)
        feats = feats.view(feats.size(0), -1)   # (num_slices, 64)
        subject_feat = feats.mean(dim=0)        # SUBJECT-LEVEL
        return self.classifier(subject_feat)    # (1)

# ---------------- EVAL ----------------

def evaluate_test_accuracy(model):
    y_true, y_pred = [], []

    for label, cls in enumerate(["CN", "AD"]):
        folder = os.path.join(TEST_DIR, cls)
        for f in os.listdir(folder):
            volume = np.load(os.path.join(folder, f))

            mean = volume.mean()
            std = volume.std() + 1e-6
            volume = (volume - mean) / std

            slices = volume.transpose(2, 0, 1)
            slices = torch.tensor(slices).float().unsqueeze(1).to(DEVICE)

            with torch.no_grad():
                logit = model(slices)
                prob = torch.sigmoid(logit).item()

            pred = 1 if prob >= 0.5 else 0
            y_true.append(label)
            y_pred.append(pred)

    return accuracy_score(y_true, y_pred)

# ---------------- MAIN ----------------

def predict(npy_path):
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    volume = np.load(npy_path)
    mean = volume.mean()
    std = volume.std() + 1e-6
    volume = (volume - mean) / std

    slices = volume.transpose(2, 0, 1)
    slices = torch.tensor(slices).float().unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        logit = model(slices)
        prob = torch.sigmoid(logit).item()

    label = "AD" if prob >= 0.5 else "CN"
    acc = evaluate_test_accuracy(model)

    print("\n--- Binary MRI Prediction ---")
    print("Prediction:", label)
    print("Confidence:", round(prob, 4))
    print("Test Accuracy:", round(acc * 100, 2), "%")

if __name__ == "__main__":
    path = input("Enter path to .npy MRI file: ")
    predict(path)

#data/processed/test/AD/136_S_0426.npy