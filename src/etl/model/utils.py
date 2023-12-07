import numpy as np
import torch
from abc import abstractmethod, ABC


class Classifier(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def predict(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        ...
