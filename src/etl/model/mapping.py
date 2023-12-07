from etl.model.deep_learning_models import ShallowClassifier, get_shallow_transforms

MODEL_NAME_TO_MODEL_INFOS = {
    "shallow": {"model": ShallowClassifier, "transforms": get_shallow_transforms}
}
