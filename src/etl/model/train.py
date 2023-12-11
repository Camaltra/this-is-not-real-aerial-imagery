import torch
from argparse import ArgumentParser
from etl.model.mapping import MODEL_NAME_TO_MODEL_INFOS
from etl.model.engine.data_classes import OptimizerParameters, ExperiementParameters
from etl.model.engine.trainers import BCTrainer
from etl.model.engine.utils import get_loaders
from etl.model.engine.tracker import ExperiementTracker
from etl.utils import parser

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--learning-rate",
        help="Learning Rate | Default=1e-4",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch Size | Default=64",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num-epoch",
        help="Number of epoch",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--model-name",
        help="The model identifiant architecture -- see /src/etl/model/mapping.py",
        type=str,
        required=True,
        choices=list(MODEL_NAME_TO_MODEL_INFOS.keys())
    )
    parser.add_argument(
        "--experiement-id",
        help="The model experiement name for registry traking",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--optim-name",
        help="The otpimizer name -- see available /src/etl/model/data_classes.py",
        type=str,
        choices=["adam", "rmsprop", "sgd"],
        required=True
    )
    parser.add_argument(
        "--adam-beta-1",
        help="Adam beta 1 optimizer hyperparameter | Default=0.9",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam-beta-2",
        help="Adam beta 2 optimizer hyperparameter | Default=0.999",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--momentum",
        help="Momentum hyperparameter for SGD | Default=0.0",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--rms-alpha",
        help="Alpha hyperparameter for RMSProp | Default=0.99",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--rms-eps",
        help="EPS Hyperparameter for RMSProp | Default=1e-8",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--weight-decay",
        help="Useless | Not Implemented yet",
    )

    return parser


@parser("Train model", "Train a single model given script parameters", parse_arguments)
def main(command_line_args):
    model_infos = MODEL_NAME_TO_MODEL_INFOS.get(command_line_args.model_name)
    trn_trms, valid_trms = model_infos.get("transforms")()

    optim = OptimizerParameters(
        command_line_args.optim_name,
        learning_rate=command_line_args.learning_rate,
        rms_alpha=command_line_args.rms_alpha,
        rms_eps=command_line_args.rms_eps,
        momentum=command_line_args.momentum,
        adam_beta_1=command_line_args.adam_beta_1,
        adam_beta_2=command_line_args.adam_beta_2,
    )
    params = ExperiementParameters(
        command_line_args.model_name,
        command_line_args.num_epoch,
        command_line_args.batch_size,
        optim,
    )

    trainer = BCTrainer(params, get_loaders(trn_trms, valid_trms, params.batch_size))

    with ExperiementTracker(command_line_args.experiement_id) as tracker:
        tracker.record_training(trainer)


if __name__ == "__main__":
    main()
