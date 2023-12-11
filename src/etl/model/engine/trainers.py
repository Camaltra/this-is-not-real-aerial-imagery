import logging

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from abc import abstractmethod, ABC
from etl.model.mapping import MODEL_NAME_TO_MODEL_INFOS
from etl.model.engine.utils import get_loaders
from etl.model.engine.data_classes import (
    ExperiementParameters,
    Metrics,
    OptimizerParameters,
)
from etl.model.engine.metrics_computor import BCMetrics
from etl.model.engine.utils import get_device, Verbose


logging.basicConfig(level=logging.INFO)


class Trainer(ABC):
    def __init__(
        self,
        training_parameters: ExperiementParameters,
        loaders: tuple[DataLoader, DataLoader],
        verbose: int | Verbose = Verbose.TRAIN_AND_VALID_METRICS,
    ):
        self.device = get_device()

        assert (
            len(loaders) == 2
        ), "There is less or more than 2 loaders, only training and valid loaders are expected"

        self.__training_parameters = training_parameters
        self._trn_loader, self._val_loader = loaders
        if isinstance(verbose, int):
            self._verbose = Verbose(verbose)
        else:
            self._verbose = verbose
        self.logger = logging.getLogger("Trainer")

        model_infos = MODEL_NAME_TO_MODEL_INFOS.get(
            self.__training_parameters.model_name
        )
        self.model = model_infos.get("model")().to(self.device)
        self.train_transform, self.valid_transform = model_infos.get("transforms")()

        self.logger = logging.getLogger("Trainer")
        # TODO 03: Display model paramter at initilisation

        self.batch_size = self.__training_parameters.batch_size
        self.num_epoch = self.__training_parameters.num_epoch

        self.loss_fn: None | nn.Module = None
        self.optimizer = self.__training_parameters.optim.build_optimizer(
            self.model.parameters()
        )

        self._training_history: list[Metrics] = []

    def display_training_paramters(self):
        print(self.__training_parameters)

    @property
    def training_paramters(self):
        return self.__training_parameters

    def save(self, filepath: str, filename: str, tag: str):
        torch.save(self.model, f"{filepath}/{filename}:{tag}.pt")

    @property
    def training_history(self):
        return self._training_history

    @abstractmethod
    def fit(self):
        pass


class BCTrainer(Trainer):
    def __init__(
        self,
        training_parameters: ExperiementParameters,
        loaders: tuple[DataLoader, DataLoader],
        verbose: bool = True,
    ):
        super().__init__(training_parameters, loaders, verbose)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(self):
        assert (
            self.loss_fn is not None
        ), f"Loss function is not set for the class {self.__class__.__name__}"

        for epoch in range(self.num_epoch):
            train_loss = self._train_fn()

            valid_loss = BCMetrics.compute_loss(
                self._val_loader, self.model, self.loss_fn, device=self.device
            )
            valid_accuracy = BCMetrics.compute_accuracy(
                self._val_loader, self.model, device=self.device
            )

            epoch_metrics = Metrics(
                train_loss=train_loss,
                valid_loss=valid_loss,
                valid_scores_metrics={"accuracy": valid_accuracy},
            )
            self._training_history.append(epoch_metrics)
            if self._verbose == Verbose.TRAIN_AND_VALID_METRICS:
                self.logger.info(epoch_metrics)

    def _train_fn(
        self,
    ) -> float:
        loop = tqdm(self._trn_loader)
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(self.device)
            targets = targets.float().to(self.device)

            predictions = self.model(data)
            loss = self.loss_fn(predictions, targets)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loop.set_postfix(loss=loss.item())

        return total_loss / len(self._trn_loader)


if __name__ == "__main__":
    optim = OptimizerParameters("adam", learning_rate=1e-4)
    params = ExperiementParameters("shallow", 1, 64, optim)

    trn_trms, valid_trms = MODEL_NAME_TO_MODEL_INFOS.get("shallow").get("transforms")()

    trainer = BCTrainer(params, get_loaders(trn_trms, valid_trms, params.batch_size))
    trainer.display_training_paramters()

    trainer.fit()
