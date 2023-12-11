import json
from argparse import ArgumentParser, Namespace
from utils import parser, ModelConfig, RecorderConfig
from pathlib import Path
from exception import (
    ExpectedClassfierVersionDoesNotExist,
    UnfoundClassifier,
    RegistryDoesNotExist,
)
import os
import re
import logging
from recorder import EarthRecorder
from dataclasses import asdict
import time

LOCAL_MODEL_REGISTRY_PATH = Path().absolute() / "model" / "registry"

logging.basicConfig(level=logging.INFO)


def parse_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--number-batch",
        help="The number of batch to collect dat -- corrolated to batch-size to get the number of sample needed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--image-collection-offset",
        help="The number of time arrow key will be hit to move the screen | Default=30",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--screenshot-width",
        help="The screenshot width | Default=512",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--screenshot-height",
        help="The screenshot height | Default=512",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--batch-size",
        help="The number of image in a batch -- corrolated to number-batch to get the number of sample needed| Default=64",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--classifier-model-tag",
        help="The tag of the classifier to use | Take the lastest by default if exist",
        type=int,
    )
    parser.add_argument(
        "--use-classifier",
        help="Decide use the ocean classifier to keep only terrestrial picture if exist | Default=True",
        action="store_true",
    )
    parser.add_argument(
        "--delete-intermediate-saves",
        help="Delete the intermediate saves from the programs | Defautl=True",
        type=bool,
        default=True,
    )

    return parser


def get_latest_file_with_regex(directory, pattern):
    matching_files = [file for file in os.listdir(directory) if re.match(pattern, file)]
    if matching_files:
        latest_file = max(
            matching_files, key=lambda f: os.path.getctime(os.path.join(directory, f))
        )
        return os.path.join(directory, latest_file)
    else:
        return None


def build_configs(
    command_line_args: Namespace,
) -> tuple[RecorderConfig, ModelConfig | None]:
    model_config = None
    if command_line_args.use_classifier:
        if not LOCAL_MODEL_REGISTRY_PATH.is_dir():
            raise RegistryDoesNotExist
        if tag := command_line_args.classifier_model_tag is not None:
            if not (LOCAL_MODEL_REGISTRY_PATH / f"classifier{tag}.pt").is_file():
                raise ExpectedClassfierVersionDoesNotExist
            model_path = (LOCAL_MODEL_REGISTRY_PATH / f"classifier{tag}.pt").is_file()
        else:
            model_path = get_latest_file_with_regex(
                LOCAL_MODEL_REGISTRY_PATH, r".*:([0-9]{2}).pt"
            )
            if model_path is None:
                raise UnfoundClassifier

        model_version = tag or re.search(r":([0-9]{2})", model_path).group()  # type: ignore
        with open(
            LOCAL_MODEL_REGISTRY_PATH / f"classifier_metadata{model_version}.json", "r"
        ) as f:
            model_performance = json.load(f)

        model_config = ModelConfig(
            model_path=model_path,
            model_version=model_version,
            accuracy=model_performance.get("accuracy", "Accuracy data unvailable"),
            train_date=model_performance.get(
                "train_date", "Train date data unvailable"
            ),
        )

    recorder_config = RecorderConfig(
        num_of_batch_to_collect=command_line_args.number_batch,
        offset=command_line_args.image_collection_offset,
        screenshot_width=command_line_args.screenshot_width,
        screenshot_height=command_line_args.screenshot_height,
        batch_save_size=command_line_args.batch_size,
        delete_intermediate_saves=command_line_args.delete_intermediate_saves,
    )

    return recorder_config, model_config


@parser(
    "ETL on Google Earth web interface",
    "Screen image from spacial area from the Google Earth interface",
    parse_arguments,
)
def main(command_line_args: Namespace) -> None:
    recorder_config, model_config = build_configs(command_line_args)
    logging.info(recorder_config)
    # if model_config:
    #     logging.info(model_config)
    recorder = EarthRecorder(**asdict(recorder_config))
    time.sleep(3)
    recorder.record()


if __name__ == "__main__":
    main()
