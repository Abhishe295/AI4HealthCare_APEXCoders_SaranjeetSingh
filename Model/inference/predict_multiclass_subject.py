import os, numpy as np, torch, torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "multiclass_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["CN", "MCI", "AD"]

# -------- MODEL (same as training) --------

class SubjectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
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
        subject_feat = f.mean(dim=0)
        return self.classifier(subject_feat)

# -------- PREDICT --------

def predict(path):
    model = SubjectCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    volume = np.load(path)
    volume = (volume - volume.mean()) / (volume.std() + 1e-6)

    x = torch.tensor(volume.transpose(2,0,1)).float().unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=0).cpu().numpy()

    print("\n--- MULTICLASS MRI PREDICTION ---")
    print("Prediction:", CLASSES[probs.argmax()])
    print("Probabilities:")
    for c,p in zip(CLASSES, probs):
        print(f"  {c}: {p:.3f}")

if __name__ == "__main__":
    p = input("Path to .npy MRI file: ")
    predict(p)
