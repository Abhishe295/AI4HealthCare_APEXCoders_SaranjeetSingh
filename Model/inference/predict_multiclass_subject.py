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
MODEL_PATH = os.path.join(BASE, "models", "multiclass_model.pth")
TEST_DIR = os.path.join(BASE, "data", "processed", "test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["CN", "MCI", "AD"]

# ---------------- MODEL ----------------

class SubjectCNN(nn.Module):
    def __init__(self):
        super().__init__()
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

        # 🔥 ATTENTION POOLING (MUST EXIST)
        self.attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # x: (num_slices, 1, H, W)
        feats = self.encoder(x).view(x.size(0), -1)   # (S,128)

        attn_scores = self.attn(feats)                # (S,1)
        attn_weights = torch.softmax(attn_scores, dim=0)

        subject_feat = (feats * attn_weights).sum(dim=0)

        return self.classifier(subject_feat)


# ---------------- EVAL ----------------

def evaluate():
    model = SubjectCNN().to(DEVICE)
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
                logits = model(x)
                probs = torch.softmax(logits, dim=0).cpu().numpy()

            y_true.append(label)
            y_pred.append(probs.argmax())
            y_prob.append(probs)

    # ---------- METRICS ----------
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    f1 = f1_score(y_true, y_pred, average="macro")

    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2]
    )

    print("\n--- MULTICLASS TEST METRICS ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"ROC-AUC (OvR): {auc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    for i,c in enumerate(CLASSES):
        print(f"{c} → Precision: {prec[i]:.3f}, Recall: {rec[i]:.3f}")

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot()
    plt.title("Multiclass Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate()
