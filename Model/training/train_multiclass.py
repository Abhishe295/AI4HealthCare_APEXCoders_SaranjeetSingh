import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import MRIDataset
from model import MRIModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10
LR = 1e-4


def train():
    model = MRIModel(3).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(MRIDataset("train"), batch_size=32, shuffle=True)
    test_loader = DataLoader(MRIDataset("test"), batch_size=32)

    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)

                logits = model(x)
                pred = logits.argmax(dim=1).cpu()

                y_true.extend(y.numpy())
                y_pred.extend(pred.numpy())

        acc = accuracy_score(y_true, y_pred) * 100
        print(f"TEST ACC: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/multiclass_model.pth")
            print("✅ Saved best")

    print("🔥 BEST:", best_acc)


if __name__ == "__main__":
    train()