import torch.nn as nn
import torchvision.models as models


class BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x)