import torch

from torch.utils.data import Dataset
from glob import glob
from random import shuffle
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import os


class EarthPicDataset(Dataset):
    def __init__(self, transform: None | A.Compose = None) -> None:
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        earth = glob(f"{current_folder_path}/data/earth/*.png")
        ocean = glob(f"{current_folder_path}/data/ocean/*.png")
        self.fpaths = earth + ocean

        self.transform: A.Compose | None = transform or A.Compose(
            [
                ToTensorV2(),
            ]
        )

        shuffle(self.fpaths)
        self.targets = [fpaths.split("/")[-2] == "earth" for fpaths in self.fpaths]

    def __getitem__(self, ix):
        img = np.array(Image.open(f"{self.fpaths[ix]}").convert("RGB"))
        target = self.targets[ix]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img.float(), torch.tensor([target]).float()

    def __len__(self):
        return len(self.fpaths)
