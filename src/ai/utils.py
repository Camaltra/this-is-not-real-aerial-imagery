import torch
from typing import Any
from torch.utils.data import DataLoader


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(img: torch.Tensor) -> torch.Tensor:
    return (img + 1) * 0.5


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def default(val: Any, default_val: Any) -> Any:
    return val if val is not None else default_val


def cycle(dl: DataLoader) -> torch.Tensor:
    while True:
        for data in dl:
            yield data


def extract(
    constants: torch.Tensor, timestamps: torch.Tensor, shape: int
) -> torch.Tensor:
    batch_size = timestamps.shape[0]
    out = constants.gather(-1, timestamps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timestamps.device)
