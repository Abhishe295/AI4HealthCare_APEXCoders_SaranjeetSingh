import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 🔥 MUST MATCH TRAINING EXACTLY
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
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        subject_feat = feats.mean(dim=0)
        return self.classifier(subject_feat)

def load_binary_model(path):
    model = BinaryCNN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def predict_binary(model, volume):
    volume = (volume - volume.mean()) / (volume.std() + 1e-6)
    x = torch.tensor(volume.transpose(2, 0, 1)).float().unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    return {
        "label": "AD" if prob >= 0.5 else "CN",
        "confidence": float(round(prob, 4))
    }

