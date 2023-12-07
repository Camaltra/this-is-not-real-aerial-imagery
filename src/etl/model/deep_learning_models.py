import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_shallow_transforms():
    train_transforms = A.Compose(
        [
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    valid_transforms = A.Compose(
        [
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    return train_transforms, valid_transforms


class ShallowClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        width = 512
        height = 512
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (width // 2) * (height // 2), 64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


class LeNet(nn.Module):
    """
    Not suppose to take as input big images
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=250000, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        return self.model(x)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Inception3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


if __name__ == "__main__":
    x = torch.rand(size=(1, 3, 512, 512))
    model = ShallowClassifier()
    print(model(x).shape)
