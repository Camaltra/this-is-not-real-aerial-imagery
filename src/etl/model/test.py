from etl.model.engine.trainers import (
    BCTrainer,
    OptimizerParameters,
    ExperiementParameters,
    MODEL_NAME_TO_MODEL_INFOS,
    get_loaders,
)
from etl.model.engine.tracker import ExperiementTracker

if __name__ == "__main__":
    optim = OptimizerParameters("adam", learning_rate=1e-4)
    optim_2 = OptimizerParameters("sgd", learning_rate=1e-3, momentum=0.4)
    params = ExperiementParameters("shallow", 1, 64, optim)
    params_2 = ExperiementParameters("shallow", 1, 128, optim_2)

    trn_trms, valid_trms = MODEL_NAME_TO_MODEL_INFOS.get("shallow").get("transforms")()

    trainer = BCTrainer(params, get_loaders(trn_trms, valid_trms, params.batch_size))
    trainer_2 = BCTrainer(
        params_2, get_loaders(trn_trms, valid_trms, params_2.batch_size)
    )
    with ExperiementTracker("test_2") as tracker:
        for i in [trainer, trainer_2]:
            tracker.record_training(i)
