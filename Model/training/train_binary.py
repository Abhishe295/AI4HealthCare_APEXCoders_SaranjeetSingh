import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset_binary import MRIBinaryDataset
from model_binary import BinaryModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 5
LR = 1e-4


def train():
    model = BinaryModel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(MRIBinaryDataset("train"), batch_size=32, shuffle=True)
    test_loader = DataLoader(MRIBinaryDataset("test"), batch_size=32)

    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.float().to(DEVICE)

            logits = model(x).squeeze()
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

                prob = torch.sigmoid(model(x).squeeze()).cpu()
                pred = (prob > 0.5).int()

                y_true.extend(y.numpy())
                y_pred.extend(pred.numpy())

        acc = accuracy_score(y_true, y_pred) * 100
        print(f"TEST ACC: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/binary_model.pth")
            print("✅ Saved best")

    print("🔥 BEST:", best_acc)


if __name__ == "__main__":
    train()