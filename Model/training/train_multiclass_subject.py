import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["CN", "MCI", "AD"]

EPOCHS = 15
LR = 1e-4

# ---------------- DATASET ----------------

class MRIDataset(Dataset):
    def __init__(self, split):
        self.samples = []
        for i, c in enumerate(CLASSES):
            d = os.path.join(DATA, split, c)
            for f in os.listdir(d):
                self.samples.append((os.path.join(d, f), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vol = np.load(path)

        # subject-wise normalization
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)

        # (H,W,D) → (D,H,W) → (D,1,H,W)
        x = torch.tensor(
            vol.transpose(2, 0, 1),
            dtype=torch.float32
        ).unsqueeze(1)

        return x, label

# ---------------- MODEL ----------------

class SubjectCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Slice encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        # 🔥 Attention network
        self.attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Subject classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # x: (num_slices, 1, H, W)

        feats = self.encoder(x)                 # (S,128,1,1)
        feats = feats.view(feats.size(0), -1)   # (S,128)

        # attention weights
        attn_scores = self.attn(feats)          # (S,1)
        attn_weights = torch.softmax(attn_scores, dim=0)

        # weighted sum → subject feature
        subject_feat = (feats * attn_weights).sum(dim=0)

        return self.classifier(subject_feat)

# ---------------- TRAIN ----------------

def train():
    model = SubjectCNN().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )

    # class imbalance handling
    class_weights = torch.tensor([1.5, 1.0, 1.8]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        MRIDataset("train"), batch_size=1, shuffle=True
    )
    val_loader = DataLoader(
        MRIDataset("val"), batch_size=1
    )

    for epoch in range(EPOCHS):
        # -------- TRAIN --------
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = x.squeeze(0).to(DEVICE)
            y = torch.tensor([y]).to(DEVICE)

            logits = model(x).unsqueeze(0)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- VALIDATE --------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.squeeze(0).to(DEVICE)
                logits = model(x)
                pred = torch.softmax(logits, dim=0).argmax().item()

                y_true.append(y.item())
                y_pred.append(pred)

        acc = accuracy_score(y_true, y_pred) * 100
        # print(f"VAL ACC: {acc:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "multiclass_model.pth")
    torch.save(model.state_dict(), save_path)
    print("✅ Model saved to:", save_path)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    train()
