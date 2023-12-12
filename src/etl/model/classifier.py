import torch
from etl.model.mapping import MODEL_NAME_TO_MODEL_INFOS
from torch.nn.functional import sigmoid
import numpy as np


class Classifier:
    def __init__(self, model_type: str, model_path):
        self._model = torch.load(model_path).to("cpu")
        _, self._valid_trasforms = MODEL_NAME_TO_MODEL_INFOS.get(model_type).get(
            "transforms"
        )()

    def predict(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if len(x.shape) == 3:
            x = self._valid_trasforms(image=x)["image"].to("cpu")
            x = x[None, :, :, :]
        else:
            transformed_images: list[torch.Tensor] = []
            for image in x:
                # Apply the transformations to each image
                transformed_image = self._valid_trasforms(image=image)["image"].to(
                    "cpu"
                )
                transformed_images.append(transformed_image)
            x = torch.stack(transformed_images)
        return sigmoid(self._model(x)).detach().numpy().reshape(-1)
