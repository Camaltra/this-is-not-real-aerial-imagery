import torch
from ai.diffusion_process import DiffusionModel
from ai.trainer import Trainer
from ai.utils import parser
from argparse import ArgumentParser, Namespace
import os
import json
from ai.mapping import MODEL_NAME_MAPPING


def parse_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-c",
        "--config-filepath",
        help="The config filepath for the model/trainer config (Litteral filepath form this file)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-milestone",
        help="The milestone of the model",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--is-old-model",
        help="If the model been trained on the old methods",
        action="store_true",
    )

    return parser


@parser(
    "Diffusion model training script",
    "Diffusion model training script",
    parse_arguments,
)
def main(command_line_args: Namespace) -> None:
    torch.mps.empty_cache()

    if not os.path.isfile(command_line_args.config_filepath):
        raise FileNotFoundError(command_line_args.config_filepath)

    with open(command_line_args.config_filepath, "r") as f:
        config_file = json.load(f)

    unet_config = config_file.get("unet_config")
    trainer_config = config_file.get("trainer_config")
    diffusion_config = config_file.get("diffusion_config")

    unet_ = MODEL_NAME_MAPPING.get(unet_config.get("model_mapping"))

    model = unet_(
        dim=unet_config.get("input"),
        channels=unet_config.get("channels"),
        dim_mults=tuple(unet_config.get("dim_mults")),
    ).to("mps")

    diffusion_model = DiffusionModel(
        model,
        image_size=diffusion_config.get("image_size"),
        beta_scheduler=diffusion_config.get("betas_scheduler"),
        timesteps=diffusion_config.get("timesteps"),
    )

    trainer = Trainer(
        diffusion_model=diffusion_model,
        folder="../data/training",
        results_folder=f'./results/{config_file.get("model_name")}',
        train_batch_size=trainer_config.get("train_batch_size"),
        train_lr=trainer_config.get("train_lr"),
        train_num_steps=trainer_config.get("train_num_steps"),
        save_and_sample_every=trainer_config.get("save_and_sample_every"),
        num_samples=trainer_config.get("num_samples"),
    )

    if milestone := command_line_args.model_milestone:
        trainer.load(milestone, is_old_model=command_line_args.is_old_model)

    trainer.train()


if __name__ == "__main__":
    main()
