import torch.nn as nn
import torchvision.models as models


class MRIModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # change input channel
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)