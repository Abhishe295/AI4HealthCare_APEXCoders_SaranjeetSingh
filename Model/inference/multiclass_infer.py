import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["CN", "MCI", "AD"]

class SubjectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 🔥 MUST MATCH TRAINING
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        f = self.encoder(x).view(x.size(0), -1)
        k = max(5, f.shape[0] // 8)
        subject_feat = torch.topk(f, k=k, dim=0).values.mean(dim=0)
        return self.classifier(subject_feat)

def load_multiclass_model(path):
    model = SubjectCNN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def predict_multiclass(model, volume):
    volume = (volume - volume.mean()) / (volume.std() + 1e-6)
    x = torch.tensor(volume.transpose(2, 0, 1)).float().unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=0).cpu().numpy()

    return {
        "label": CLASSES[int(probs.argmax())],
        "probabilities": dict(zip(CLASSES, probs.round(4)))
    }
