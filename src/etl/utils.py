from dataclasses import dataclass
from enum import Enum
from argparse import ArgumentParser
from typing import Callable
from etl.model.classifier import Classifier
from etl.recorder_strategies import RecorderStrategy


class ImageType(Enum):
    NUMPY = "np.array"
    PIL = "pillow"



@dataclass()
class ModelConfig:
    model_name: str
    model_path: str
    model_version: str
    accuracy: float | str

    def __str__(self):
        return f"Model CONFIG\n\tModel Name: {self.model_name}\n\tModel Path: {self.model_path}\n\tModel Version: {self.model_version}\n\tModel Accuracy: {f'{self.accuracy:.2f} %' if isinstance(self.accuracy, float) else self.accuracy}"


@dataclass()
class RecorderConfig:
    num_of_batch_to_collect: int
    offset: int
    screenshot_width: int
    screenshot_height: int
    batch_save_size: int
    classifier: Classifier | None
    recorder_stategy: RecorderStrategy | None
    delete_intermediate_saves: bool = True

    def __str__(self):
        return f"Recoder CONFIG:\n\tNum Of Batch To Collect: {self.num_of_batch_to_collect}\n\tOffset: {self.offset}\n\tScreenshot Width: {self.screenshot_width}\n\tScreenshot Height: {self.screenshot_height}\n\tBatch Save Size: {self.batch_save_size}\n\tDelete Intermediate Saves: {self.delete_intermediate_saves}"


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
