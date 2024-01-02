from ai.models.u_net import OneResUNet, TwoResUNet, AttentionUNet

MODEL_NAME_MAPPING = {
    "OneResUNet": OneResUNet,
    "TwoResUNet": TwoResUNet,
    "AttentionUNet": AttentionUNet,
}
