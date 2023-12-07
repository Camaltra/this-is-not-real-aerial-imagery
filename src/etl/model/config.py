from etl.model.engine.data_classes import OptimizerParameters, ExperiementParameters

TRAINING_CONFIG = [
    ExperiementParameters("shallow", 1, 64, OptimizerParameters("sgd", learning_rate=1e-3, momentum=0.4))
]