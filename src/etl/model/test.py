from etl.model.engine.trainers import (
    MODEL_NAME_TO_MODEL_INFOS,
)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
import random
from PIL import Image
import numpy as np

if __name__ == "__main__":
    trn_trms, valid_trms = MODEL_NAME_TO_MODEL_INFOS.get("pretrainedvgg16").get(
        "transforms"
    )()
    valid_path = glob("./data/valid/earth/*.png") + glob("./data/valid/ocean/*.png")
    random.shuffle(valid_path)

    model = torch.load("./registry/vgg16pretrained/classfier:09.pt").to("mps")
    for img_path in valid_path[:100]:
        img = np.array(Image.open(img_path).convert("RGB"))
        trms_img = valid_trms(image=img).get("image").to("mps")[None, :, :, :]
        print(trms_img.shape)
        pred = model(trms_img)
        plt.imshow(img)
        plt.title(
            f"Preds: {'earth' if nn.functional.sigmoid(pred).item() > 0.5 else 'ocean'}"
        )
        plt.show()
