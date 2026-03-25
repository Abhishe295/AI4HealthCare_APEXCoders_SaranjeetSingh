import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 🔥 MODEL (same as training)
class BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x)


def load_binary_model(path):
    model = BinaryModel().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# 🔥 TRANSFORM (IMPORTANT — SAME AS TRAINING)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def predict_binary(model, image):
    img = Image.open(image)
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()

    return {
        "label": "AD" if prob > 0.5 else "CN",
        "confidence": float(round(prob, 4))
    }