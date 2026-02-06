import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15
BATCH_SIZE = 1
LR = 1e-4

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
        volume = np.load(path)

        mean = volume.mean()
        std = volume.std() + 1e-6
        volume = (volume - mean) / std
          # (128,128,64)
        slices = volume.transpose(2, 0, 1)  # (64,128,128)
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
        # x: (num_slices, 1, H, W)
        feats = self.features(x)          # (num_slices, 64, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (num_slices, 64)

        subject_feat = feats.mean(dim=0)  # ✅ SUBJECT-LEVEL
        logit = self.classifier(subject_feat)  # (1)

        return logit


# ---------------- TRAINING ----------------

def train():
    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([32/23]).to(DEVICE)
)


    train_loader = DataLoader(MRIDataset("train"), batch_size=1, shuffle=True)
    val_loader = DataLoader(MRIDataset("val"), batch_size=1)

    for epoch in range(EPOCHS):
        model.train()
        for slices, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            slices = slices.squeeze(0).to(DEVICE)
            label = torch.tensor([label], dtype=torch.float32).to(DEVICE)

            logit = model(slices)          # single logit
            loss = criterion(logit.view(1), label)




            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} complete")

    save_path = os.path.join(MODEL_DIR, "binary_model.pth")
    torch.save(model.state_dict(), save_path)

    print("Model saved to:", save_path)
    print("File size (bytes):", os.path.getsize(save_path))


if __name__ == "__main__":
    train()


# data/processed/test/AD/136_S_0426.npy

