from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from glob import glob


class AerialDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        augment_horizontal_flip=False,
    ):
        self.paths = [path for path in glob(f"{folder}/*.png")]
        self.image_size = image_size

        self.transform = T.Compose(
            [
                T.Resize(image_size),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        path = self.paths[ix]
        img = Image.open(path).convert("RGB")
        return self.transform(img)
