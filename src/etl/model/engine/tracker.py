import json
import logging
import os.path

import torch
import torch.nn as nn

from etl.model.engine.data_classes import (
    Metrics,
    ExperiementParameters,
    ModelExperiment,
    ExperimentRegistry,
    DetailedModelExperiment,
)
from etl.model.engine.trainers import Trainer
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class ExperiementTracker:
    TRACKING_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "registry"
    TRACK_RECORD_FILENAME = "registry_logs.json"

    def __init__(self, experiment_id: str) -> None:
        self.logger = logging.getLogger("ExperiementTracker")
        self.experiment_id = experiment_id

    def __enter__(self):
        self.logger.info("Oppening tracking for ID <%s>", self.experiment_id)
        if not os.path.isdir(self.TRACKING_PATH / self.experiment_id):
            os.mkdir(self.TRACKING_PATH / self.experiment_id)
        self.tracking = self._get_trackings()
        return self

    def _get_trackings(self):
        if os.path.isfile(
            self.TRACKING_PATH / self.experiment_id / self.TRACK_RECORD_FILENAME
        ):
            with open(
                self.TRACKING_PATH / self.experiment_id / self.TRACK_RECORD_FILENAME,
                "r",
            ) as f:
                records = json.load(f)
            experiment_models = [
                ModelExperiment(
                    experiment.get("model_tag"), experiment.get("model_accuracy")
                )
                for experiment in records.get("experiments")
            ]
            experiment_regitry = ExperimentRegistry(
                (records.get("experiement_number") or self.experiment_id),
                records.get("last_tag_processed"),
                records.get("best_model_tag"),
                records.get("best_model_accuracy"),
                experiment_models,
            )
        else:
            experiment_regitry = ExperimentRegistry()
        return experiment_regitry

    def __exit__(self, exc_type, exc_value, tb):
        self._save_tracking(self.tracking)
        self.logger.info("Closing tracking for ID <%s>", self.experiment_id)

    def record_training(self, trainer: Trainer):
        detailled_metrics = None
        try:
            self.logger.info(
                "Start training experiement:\n%s", trainer.training_paramters
            )
            trainer.fit()
            training_history = trainer.training_history
            detailled_metrics = self._build_detailled_metrics(
                training_history, trainer.training_paramters
            )
            self._save_model_and_detailled_metrics(detailled_metrics, trainer.model)
            self.logger.info("End training experiment, see data in the registry")
        except Exception as e:
            self.logger.error(
                "Unkown error happen during the fit of the config:\n%s",
                trainer.training_paramters,
            )
            self.logger.debug(e)
        if detailled_metrics is not None:
            self.tracking.last_tag_processed = detailled_metrics.model_tag
            if detailled_metrics.model_accuracy > (
                self.tracking.best_model_accuracy or 0.0
            ):
                self.tracking.best_model_accuracy = detailled_metrics.model_accuracy
                self.tracking.best_model_tag = detailled_metrics.model_tag
            self.tracking.experiments.append(
                ModelExperiment(
                    detailled_metrics.model_tag, detailled_metrics.model_accuracy
                )
            )

    def _save_model_and_detailled_metrics(
        self, detailled_metrics: DetailedModelExperiment, model: nn.Module
    ) -> None:
        filepath = self.TRACKING_PATH / self.experiment_id
        detailled_metrics_filename = (
            f"classfier_metadata:{detailled_metrics.model_tag}.json"
        )
        with open(filepath / detailled_metrics_filename, "w") as f:
            json.dump(asdict(detailled_metrics), f, indent=2)
        torch.save(model, filepath / f"classfier:{detailled_metrics.model_tag}.pt")
        self.logger.info("Model and experiements history saved...")

    def _build_detailled_metrics(
        self, training_history: list[Metrics], model_parameters: ExperiementParameters
    ):
        model_tag = self._get_next_model_tag()
        return DetailedModelExperiment(
            model_tag=model_tag,
            model_accuracy=training_history[-1].valid_scores_metrics.get("accuracy"),
            model_train_loss_history=[hst.train_loss for hst in training_history],
            model_valid_loss_history=[hst.valid_loss for hst in training_history],
            model_valid_accuracy_history=[
                hst.valid_scores_metrics.get("accuracy") for hst in training_history
            ],
            experiment_parameters=model_parameters,
        )

    def _get_next_model_tag(self):
        if self.tracking.last_tag_processed is None:
            return "00"
        last_tag_processed = int(self.tracking.last_tag_processed)
        if last_tag_processed + 1 <= 9:
            last_tag_processed = f"0{last_tag_processed + 1}"
        else:
            last_tag_processed = str(last_tag_processed + 1)
        return last_tag_processed

    def _save_tracking(self, experiment_regitry):
        with open(
            self.TRACKING_PATH / self.experiment_id / self.TRACK_RECORD_FILENAME, "w"
        ) as f:
            json.dump(asdict(experiment_regitry), f, indent=2)


if __name__ == "__main__":
    with ExperiementTracker("3924") as e:
        print(type(e))
