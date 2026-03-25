import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "binary")

CLASSES = ["CN", "AD"]


class MRIBinaryDataset(Dataset):
    def __init__(self, split):
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        for label, cls in enumerate(CLASSES):
            folder = os.path.join(DATA_DIR, split, cls)

            for f in os.listdir(folder):
                self.samples.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path)
        img = self.transform(img)

        return img, label