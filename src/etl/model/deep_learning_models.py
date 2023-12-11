import torch
import torch.nn as nn
import albumentations as A  # type: ignore
from albumentations.pytorch import ToTensorV2  # type: ignore
from torchsummary import summary
from torchvision import models
from torchvision.models.vgg import VGG16_Weights, VGG19_Weights


def get_shallow_transforms() -> tuple[A.Compose, A.Compose]:
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
    def __init__(self) -> None:
        super().__init__()
        width = 512
        height = 512
        self._model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (width // 2) * (height // 2), 64),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


def get_lenet_transforms() -> tuple[A.Compose, A.Compose]:
    train_transforms = A.Compose(
        [
            A.Resize(32, 32),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    valid_transforms = A.Compose(
        [
            A.Resize(32, 32),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    return train_transforms, valid_transforms


class LeNet(nn.Module):
    """
    Not suppose to take as input big images
    """

    def __init__(self) -> None:
        super().__init__()
        width = 32
        height = 32
        latent_space = 120
        self._model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Flatten(),
            nn.Linear(in_features=latent_space, out_features=latent_space),
            nn.ReLU(),
            nn.Linear(in_features=latent_space, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


def get_alexnet_transforms() -> tuple[A.Compose, A.Compose]:
    train_transforms = A.Compose(
        [
            A.Resize(256, 256),
            A.CenterCrop(227, 227),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )

    valid_transforms = A.Compose(
        [
            A.Resize(256, 256),
            A.CenterCrop(227, 227),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )

    return train_transforms, valid_transforms


class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO 05: See screen to training variables
        self._model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=256, out_channels=348, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=348, out_channels=348, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=348, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


transform = A.Compose([])


def get_vgg_transforms() -> tuple[A.Compose, A.Compose]:
    train_transforms = A.Compose(
        [
            A.Resize(256, 256, interpolation=A.InterpolationMode.BILINEAR),
            A.CenterCrop(224, 224),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.48235, 0.45882, 0.40784],
                std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )

    valid_transforms = A.Compose(
        [
            A.Resize(256, 256, interpolation=A.InterpolationMode.BILINEAR),
            A.CenterCrop(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )

    return train_transforms, valid_transforms


class VGG16Block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, number_of_conv: int):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                for _ in range(number_of_conv - 1)
            ]
        )
        self.relu_layers = nn.ModuleList([nn.ReLU() for _ in range(number_of_conv - 1)])
        self.batch_norms_layers = nn.ModuleList(
            [nn.BatchNorm2d(out_channel) for _ in range(number_of_conv - 1)]
        )

        self.first_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        for conv_layer, relu_layer, batch_norm_layer in zip(
            self.conv_layers, self.relu_layers, self.batch_norms_layers
        ):
            x = conv_layer(x)
            x = batch_norm_layer(x)
            x = relu_layer(x)
        return self.max_pool(x)


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            VGG16Block(in_channel=3, out_channel=64, number_of_conv=2),
            VGG16Block(in_channel=64, out_channel=128, number_of_conv=2),
            VGG16Block(in_channel=128, out_channel=256, number_of_conv=3),
            VGG16Block(in_channel=256, out_channel=512, number_of_conv=3),
            VGG16Block(in_channel=512, out_channel=512, number_of_conv=3),
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            # Put the dropout before RELU due to computation efficiency
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            # Put the dropout before RELU due to computation efficiency
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        return self._model(x)


class PretrainedVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self._model.classifier[3] = nn.Linear(in_features=4096, out_features=4096)
        self._model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
        for name, parameters in self._model.named_parameters():
            if name.startswith("classifier.6") or name.startswith("classifier.3"):
                parameters.requires_grad = True
            else:
                parameters.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class PretrainedVGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self._model.classifier[3] = nn.Linear(in_features=4096, out_features=4096)
        self._model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
        for name, parameters in self._model.named_parameters():
            if name.startswith("classifier.6") or name.startswith("classifier.3"):
                parameters.requires_grad = True
            else:
                parameters.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


if __name__ == "__main__":
    x = torch.rand(size=(1, 3, 512, 512))
    model = ShallowClassifier()
    print(f"SHALLOW TEST OUTPUT: {model(x)}")

    x = torch.rand(size=(1, 3, 32, 32))
    model = LeNet()  # type: ignore
    print(f"LENET TEST OUTPUT: {model(x)}")

    x = torch.rand(size=(1, 3, 227, 227))
    model = AlexNet()  # type: ignore
    print(f"ALEXNET TEST OUTPUT: {model(x)}")

    x = torch.rand(size=(1, 3, 224, 224))
    model = VGG16()  # type: ignore
    print(f"VGG16 TEST OUTPUT: {model(x)}")

    x = torch.rand(size=(1, 3, 224, 224))
    model = PretrainedVGG16()  # type: ignore
    print(f"PRETRAINED VGG16 TEST OUTPUT: {model(x)}")

    x = torch.rand(size=(1, 3, 224, 224))
    model = PretrainedVGG19()  # type: ignore
    print(f"PRETRAINED VGG19 TEST OUTPUT: {model(x)}")
