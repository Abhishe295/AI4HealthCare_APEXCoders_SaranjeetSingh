import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 20
LR = 1e-4

Z_START, Z_END = 20, 44   # ROI slices
ROI_SIZE = 64             # 64x64 crop

# ---------------- DATASET ----------------

class MRIDataset(Dataset):
    def __init__(self, split):
        self.samples = []
        for label, cls in enumerate(["CN", "AD"]):
            folder = os.path.join(DATA_DIR, split, cls)
            for f in os.listdir(folder):
                self.samples.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vol = np.load(path)

        # normalize per subject
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)

        H, W, _ = vol.shape
        cx, cy = H // 2, W // 2
        r = ROI_SIZE // 2

        slices = []
        for z in range(Z_START, Z_END):
            roi = vol[cx-r:cx+r, cy-r:cy+r, z]
            slices.append(roi)

        slices = np.stack(slices)  # (num_slices, 64, 64)
        slices = torch.tensor(slices, dtype=torch.float32).unsqueeze(1)

        return slices, label

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
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        subject_feat = feats.mean(dim=0)
        return self.classifier(subject_feat)

# ---------------- TRAIN ----------------

def train():
    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.3]).to(DEVICE)
    )

    loader = DataLoader(MRIDataset("train"), batch_size=1, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for slices, label in tqdm(loader, desc=f"Epoch {epoch+1}"):
            slices = slices.squeeze(0).to(DEVICE)
            label = torch.tensor([label], dtype=torch.float32).to(DEVICE)

            logit = model(slices)
            loss = criterion(logit.view(1), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss/len(loader):.4f}")

    save_path = os.path.join(MODEL_DIR, "binary_model.pth")
    torch.save(model.state_dict(), save_path)
    print("✅ Model saved:", save_path)

if __name__ == "__main__":
    train()
