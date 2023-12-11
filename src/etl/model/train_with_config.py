import json

import torch
from argparse import ArgumentParser
from etl.model.mapping import MODEL_NAME_TO_MODEL_INFOS
from etl.model.engine.data_classes import OptimizerParameters, ExperiementParameters
from etl.model.engine.trainers import BCTrainer
from etl.model.engine.utils import get_loaders
from etl.model.engine.tracker import ExperiementTracker
from etl.utils import parser
from etl.exception import MissingConfigParameters
from etl.model.engine.trainers import Trainer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-c" "--config-filename",
        help="JSON Config in the current directory -- See ./config_template.json to get a template",
        type=str,
    )

    return parser


def build_trainers(config_trainers: list[dict]) -> list[Trainer]:
    # TODO: Create a validation data step fron the JSON config file, here it's not implemented as I only change things
    trainers: list[Trainer] = []
    for config in config_trainers:
        optimizer_parameters = config.get("optimizer_parameters")
        if optimizer_parameters is None:
            raise MissingConfigParameters("optimizer_parameters")
        otpim = OptimizerParameters(
            optimizer=optimizer_parameters.get("optimizer_name"),
            learning_rate=optimizer_parameters.get("learning_rate"),
            rms_alpha=optimizer_parameters.get("rms_alpha") or 0.99,
            rms_eps=optimizer_parameters.get("rms_eps") or 1e-8,
            momentum=optimizer_parameters.get("momentum") or 0,
            adam_beta_1=optimizer_parameters.get("adam_beta_1") or 0.9,
            adam_beta_2=optimizer_parameters.get("adam_beta_2") or 0.999,
        )

        experiment_parameters = config.get("experiment_parameters")
        if experiment_parameters is None:
            raise MissingConfigParameters("experiment_parameters")
        experiment = ExperiementParameters(
            model_name=experiment_parameters.get("model_name"),
            num_epoch=experiment_parameters.get("num_epoch"),
            batch_size=experiment_parameters.get("batch_size"),
            optim=otpim,
        )
        trn_transform, valid_transform = MODEL_NAME_TO_MODEL_INFOS.get(
            experiment_parameters.get("model_name")
        ).get("transforms")()

        trainers.append(
            BCTrainer(
                experiment,
                get_loaders(
                    trn_transform,
                    valid_transform,
                    experiment_parameters.get("batch_size"),
                ),
            )
        )

    return trainers


@parser("Train model", "Train a single model given script parameters", parse_arguments)
def main(command_line_args):
    with open(f"./{command_line_args.c__config_filename}") as f:
        training_config = json.load(f)

    experiment_id = training_config.get("experiment_id")
    if experiment_id is None:
        raise MissingConfigParameters("experiment_id")

    trainers = build_trainers(training_config.get("trainers_config"))

    with ExperiementTracker(training_config.get("experiment_id")) as tracker:
        for trainer in trainers:
            tracker.record_training(trainer)


if __name__ == "__main__":
    main()
