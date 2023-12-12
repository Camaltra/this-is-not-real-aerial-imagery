import json
from argparse import ArgumentParser, Namespace

import torch

from utils import parser, ModelConfig, RecorderConfig
from pathlib import Path
from exception import (
    ExpectedClassfierVersionDoesNotExist,
    ExperiementRequired,
    UnfoundClassifier,
    BatchSizeCantBeZeroOrNegatif,
)
import os
import re
import logging
from recorder import EarthRecorder
from dataclasses import asdict
import time
from etl.model.classifier import Classifier
import numpy as np

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
        type=str,
    )
    parser.add_argument(
        "--experiment-name",
        help="The experiement name for the model selection",
        type=str,
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


def get_experiement_from_tag(
    experiments: list[dict], tag: str
) -> dict[str, str | float] | None:
    for experiment in experiments:
        if experiment.get("model_tag") == tag:
            return experiment
    return None


def build_classifier_config(command_line_args: Namespace) -> ModelConfig:
    if command_line_args.experiment_name is None:
        raise ExperiementRequired
    registry_path = LOCAL_MODEL_REGISTRY_PATH / command_line_args.experiment_name
    with open(registry_path / "registry_logs.json", "r") as f:
        registry_logs = json.load(f)
    if command_line_args.classifier_model_tag is not None:
        tag = command_line_args.classifier_model_tag
        experiments = registry_logs.get("experiments")
        experiment = get_experiement_from_tag(experiments, tag)
        if experiment is None:
            raise ExpectedClassfierVersionDoesNotExist(tag)
    else:
        tag = registry_logs.get("best_model_tag")
        if tag is None:
            raise ExpectedClassfierVersionDoesNotExist(tag)

    print(registry_path / f"classifier:{tag}.pt")
    if not (registry_path / f"classifier:{tag}.pt").is_file():
        raise UnfoundClassifier
    with open(registry_path / f"classifier_metadata:{tag}.json", "r") as f:
        detailed_model_train_summary = json.load(f)

    model_config = ModelConfig(
        model_name=detailed_model_train_summary.get("experiment_parameters").get(
            "model_name"
        ),
        model_path=str(registry_path / f"classifier:{tag}.pt"),
        model_version=detailed_model_train_summary.get("model_tag"),
        accuracy=detailed_model_train_summary.get(
            "model_accuracy", "Accuracy data unvailable"
        ),
    )

    return model_config


def build_configs(
    command_line_args: Namespace,
) -> tuple[RecorderConfig, ModelConfig | None]:
    classifier = None
    model_config = None
    if command_line_args.use_classifier:
        model_config = build_classifier_config(command_line_args)
        classifier = Classifier(model_config.model_name, model_config.model_path)

    if command_line_args.batch_size <= 0:
        raise BatchSizeCantBeZeroOrNegatif

    recorder_config = RecorderConfig(
        num_of_batch_to_collect=command_line_args.number_batch,
        offset=command_line_args.image_collection_offset,
        screenshot_width=command_line_args.screenshot_width,
        screenshot_height=command_line_args.screenshot_height,
        batch_save_size=command_line_args.batch_size,
        classifier=classifier,
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
    if model_config:
        logging.info(model_config)

    recorder = EarthRecorder(**asdict(recorder_config))
    time.sleep(3)
    recorder.record()


if __name__ == "__main__":
    main()
