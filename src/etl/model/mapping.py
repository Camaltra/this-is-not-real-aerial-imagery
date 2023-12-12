from etl.model.deep_learning_models import (
    ShallowClassifier,
    get_shallow_transforms,
    LeNet,
    get_lenet_transforms,
    AlexNet,
    get_alexnet_transforms,
    VGG16,
    PretrainedVGG16,
    PretrainedVGG19,
    get_vgg_transforms,
)

MODEL_NAME_TO_MODEL_INFOS = {
    "shallow": {"model": ShallowClassifier, "transforms": get_shallow_transforms},
    "lenet": {"model": LeNet, "transforms": get_lenet_transforms},
    "alexnet": {"model": AlexNet, "transforms": get_alexnet_transforms},
    "vgg16": {"model": VGG16, "transforms": get_vgg_transforms},
    "pretrainedvgg16": {"model": PretrainedVGG16, "transforms": get_vgg_transforms},
    "pretrainedvgg19": {"model": PretrainedVGG19, "transforms": get_vgg_transforms},
}
