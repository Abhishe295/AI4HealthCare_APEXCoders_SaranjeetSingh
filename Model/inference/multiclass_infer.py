import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["CN", "MCI", "AD"]


class MRIModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(512, 3)

    def forward(self, x):
        return self.model(x)


def load_multiclass_model(path):
    model = MRIModel().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def predict_multiclass(model, image):
    img = Image.open(image)
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)[0].cpu().numpy()

    return {
        "label": CLASSES[int(probs.argmax())],
        "probabilities": {
            "CN": float(round(probs[0], 4)),
            "MCI": float(round(probs[1], 4)),
            "AD": float(round(probs[2], 4)),
        }
    }