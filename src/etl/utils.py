from dataclasses import dataclass
from enum import Enum
from argparse import ArgumentParser
from typing import Callable


class ImageType(Enum):
    NUMPY = "np.array"
    PIL = "pillow"


@dataclass(frozen=True)
class Coordinates:
    x: int
    y: int

    def __sub__(self, other: "Coordinates") -> tuple[int, int]:
        return self.x - other.x, self.y - other.y

    def __str__(self):
        print("Coordinates (x<%s>, y<%s>)", self.x, self.y)


@dataclass()
class ModelConfig:
    model_path: str
    model_version: str
    accuracy: float | str
    train_date: str

    def __str__(self):
        return f"Model CONFIG\n\tModel Version: {self.model_version}\n\tModel Accuracy: {f'{self.accuracy}:.2f %' if isinstance(self.accuracy, float) else self.accuracy}\n\tTrain Date: {self.train_date}"


@dataclass()
class RecorderConfig:
    num_of_batch_to_collect: int
    offset: int
    screenshot_width: int
    screenshot_height: int
    batch_save_size: int
    model_config: ModelConfig | None = None
    delete_intermediate_saves: bool = True

    def __str__(self):
        return f"Recoder CONFIG:\n\tNum Of Batch To Collect: {self.num_of_batch_to_collect}\n\tOffset: {self.offset}\n\tScreenshot Width: {self.screenshot_width}\n\tScreenshot Height: {self.screenshot_height}\n\tBatch Save Size: {self.batch_save_size}\n\tNodel Use: {self.model_config.model_version if self.model_config is not None else None}\n\tDelete Intermediate Saves: {self.delete_intermediate_saves}"


def parser(
    prog_name: str, dscr: str, get_args: Callable[[ArgumentParser], ArgumentParser]
) -> Callable[[Callable], Callable]:
    def decorator(function):
        def new_function(*args, **kwargs):
            prs = ArgumentParser(
                prog=prog_name,
                description=dscr,
            )

            prs = get_args(prs)
            args = prs.parse_args()
            function(args)

        return new_function

    return decorator
