import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "train")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_train_samples():
    samples = []
    for label, cls in enumerate(["CN", "AD"]):
        folder = os.path.join(DATA_DIR, cls)
        for f in os.listdir(folder):
            samples.append((os.path.join(folder, f), label))
    return samples

model = SimpleCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

samples = load_train_samples()

# TRAIN HARD ON TRAIN SET
for epoch in range(15):
    correct = 0
    for path, label in samples:
        volume = np.load(path)

        mean = volume.mean()
        std = volume.std() + 1e-6
        volume = (volume - mean) / std

        slices = volume.transpose(2, 0, 1)
        slices = torch.tensor(slices).float().unsqueeze(1).to(DEVICE)
        label = torch.tensor([label], dtype=torch.float32).to(DEVICE)

        logits = model(slices).mean(dim=0)
        loss = criterion(logits.view(1), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (torch.sigmoid(logits) >= 0.5).int().item()
        correct += int(pred == label.item())

    acc = correct / len(samples)
    print(f"Epoch {epoch+1} | TRAIN accuracy: {acc:.4f}")
