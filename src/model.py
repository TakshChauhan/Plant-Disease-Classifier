"""Builds a pretrained torchvision backbone with a fresh classification head.

Supports resnet18, resnet50, efficientnet_b0, mobilenet_v2 — chosen for a spread
of accuracy/speed/size trade-offs worth comparing in the README's "extensions" section.
"""
import torch.nn as nn
from torchvision import models


def get_model(model_name: str = "resnet18", num_classes: int = 38, pretrained: bool = True):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model, backbone_params


def set_backbone_trainable(backbone_params, trainable: bool) -> None:
    """Phase 1 of training freezes these; phase 2 unfreezes them for fine-tuning."""
    for p in backbone_params:
        p.requires_grad = trainable


def get_last_conv_layer(model, model_name: str):
    """Returns the last conv layer — used as the Grad-CAM target layer."""
    model_name = model_name.lower()
    if model_name in ("resnet18", "resnet50"):
        return model.layer4[-1]
    if model_name in ("efficientnet_b0", "mobilenet_v2"):
        return model.features[-1]
    raise ValueError(f"No known last-conv-layer mapping for: {model_name}")
