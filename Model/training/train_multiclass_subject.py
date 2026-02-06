import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["CN","MCI","AD"]

# ---------------- DATASET ----------------

class MRIDataset(Dataset):
    def __init__(self, split):
        self.samples = []
        for i,c in enumerate(CLASSES):
            d = os.path.join(DATA, split, c)
            for f in os.listdir(d):
                self.samples.append((os.path.join(d,f), i))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p,l = self.samples[idx]
        v = np.load(p)
        v = (v - v.mean())/(v.std()+1e-6)
        x = torch.tensor(v.transpose(2,0,1)).float().unsqueeze(1)
        return x, l

# ---------------- MODEL ----------------

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
        k = max(5, f.shape[0] // 8)
        subject_feat = torch.topk(f, k=k, dim=0).values.mean(dim=0)
     # 🔥 SUBJECT VECTOR
        return self.classifier(subject_feat)

# ---------------- TRAIN ----------------

def train():
    model = SubjectCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    class_weights = torch.tensor([1.5, 1.0, 1.8]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)


    train_loader = DataLoader(MRIDataset("train"), batch_size=1, shuffle=True)
    val_loader = DataLoader(MRIDataset("val"), batch_size=1)

    for epoch in range(15):
        model.train()
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x,y = x.squeeze(0).to(DEVICE), torch.tensor([y]).to(DEVICE)
            loss = loss_fn(model(x).unsqueeze(0), y)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x,y in val_loader:
                x = x.squeeze(0).to(DEVICE)
                p = model(x).argmax().item()
                yt.append(y.item()); yp.append(p)

        acc = accuracy_score(yt,yp)*100
        print(f"VAL ACC: {acc:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR,"multiclass_model.pth"))
    print("Model saved")

if __name__ == "__main__":
    train()
