"""
@author supermantx
@date 2024/4/26 13:14
"""
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from .mobilefacenet import get_mbf


def get_model(model_name, cfg=None):
    if model_name == "ResNet18":
        return iresnet18()
    elif model_name == "ResNet34":
        return iresnet34()
    elif model_name == "ResNet50":
        return iresnet50()
    elif model_name == "ResNet100":
        return iresnet100()
    elif model_name == "MobileNetV2":
        if cfg:
            return get_mbf(cfg.SOLVER.FP16, num_features=512, blocks=(1, 4, 6, 2), scale=2)
        return get_mbf(True, 512, blocks=(1, 4, 6, 2), scale=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")
