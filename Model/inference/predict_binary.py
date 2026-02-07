import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, f1_score,
    precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)

# ---------------- CONFIG ----------------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "binary_model.pth")
TEST_DIR = os.path.join(BASE, "data", "processed", "test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["CN", "AD"]

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
        f = self.features(x).view(x.size(0), -1)
        subject_feat = f.mean(dim=0)
        return self.classifier(subject_feat)


# ---------------- EVAL ----------------

def evaluate():
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    for label, cls in enumerate(CLASSES):
        folder = os.path.join(TEST_DIR, cls)
        for f in os.listdir(folder):
            v = np.load(os.path.join(folder, f))
            v = (v - v.mean()) / (v.std() + 1e-6)

            x = torch.tensor(v.transpose(2,0,1)).float().unsqueeze(1).to(DEVICE)

            with torch.no_grad():
                logit = model(x)
                prob = torch.sigmoid(logit).item()

            y_true.append(label)
            y_pred.append(1 if prob >= 0.5 else 0)
            y_prob.append(prob)

    # ---------- METRICS ----------
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, average="macro")

    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1]
    )

    print("\n--- BINARY TEST METRICS ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    for i,c in enumerate(CLASSES):
        print(f"{c} → Precision: {prec[i]:.3f}, Recall: {rec[i]:.3f}")

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot()
    plt.title("Binary Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate()
