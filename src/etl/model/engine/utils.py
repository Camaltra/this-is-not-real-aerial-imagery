from torch.utils.data import DataLoader
import albumentations as A
from etl.model.dataset import EarthPicDataset
import torch
from enum import Enum


class Verbose(Enum):
    TRAIN_ONLY = 1
    TRAIN_AND_VALID_METRICS = 2


def get_loaders(
    train_transform: A.Compose, valid_transform: A.Compose, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    trn_ds = EarthPicDataset(train_transform)
    val_ds = EarthPicDataset(valid_transform)
    return DataLoader(
        trn_ds, batch_size=batch_size, shuffle=True, drop_last=True
    ), DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.backends.cuda.is_built():
        return "cuda"
    else:
        return "cpu"
