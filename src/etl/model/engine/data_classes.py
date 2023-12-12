from dataclasses import dataclass, asdict, field
import torch


@dataclass()
class OptimizerParameters:
    # TODO 04: Create a parent class with optimizer
    optimizer: str
    learning_rate: float
    rms_alpha: float = 0.99
    rms_eps: float = 1e-08
    momentum: float = 0
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999

    def build_optimizer(self, model_parameters):
        if self.optimizer == "adam":
            return torch.optim.Adam(
                model_parameters,
                lr=self.learning_rate,
                betas=(self.adam_beta_1, self.adam_beta_2),
            )
        elif self.optimizer == "sgd":
            return torch.optim.SGD(
                model_parameters, lr=self.learning_rate, momentum=self.momentum
            )
        elif self.optimizer == "rmsprop":
            return torch.optim.RMSprop(
                model_parameters,
                lr=self.learning_rate,
                alpha=self.rms_alpha,
                eps=self.rms_eps,
            )

    def __str__(self):
        if self.optimizer == "adam":
            return f"OPTIM: Adam\n\t\tLearning rate: {self.learning_rate}\n\t\tBeta 1: {self.adam_beta_1}\n\t\tBeta 2: {self.adam_beta_2}"
        elif self.optimizer == "sgd":
            return f"OPTIM: SGD\n\t\tLearning rate: {self.learning_rate}\n\t\tMomentum: {self.momentum}"
        elif self.optimizer == "rmsprop":
            return f"OPTIM: RMSProp\n\t\tLearning rate: {self.learning_rate}\n\t\tAlpha: {self.rms_alpha}\n\t\tEPS: {self.rms_eps}"


@dataclass()
class ExperiementParameters:
    model_name: str
    num_epoch: int
    batch_size: int
    optim: OptimizerParameters

    def __str__(self):
        return f"EXPERIMENTS PARAMETERS:\n\tModel: {self.model_name}\n\tNum Epoch: {self.num_epoch}\n\tBatch Size: {self.batch_size}\n\t{str(self.optim)}"


@dataclass()
class ModelExperiment:
    model_tag: str
    model_accuracy: str


@dataclass()
class DetailedModelExperiment:
    model_tag: str
    model_accuracy: str
    model_train_loss_history: list[float]
    model_valid_loss_history: list[float]
    model_valid_accuracy_history: list[float]
    experiment_parameters: ExperiementParameters


@dataclass()
class ExperimentRegistry:
    experiment_number: str | None = None
    last_tag_processed: str | None = None
    best_model_tag: str | None = None
    best_model_accuracy: str | None = None
    experiments: list[ModelExperiment] = field(default_factory=list)


@dataclass()
class Metrics:
    train_loss: float
    valid_loss: float
    valid_scores_metrics: dict[str, float]
