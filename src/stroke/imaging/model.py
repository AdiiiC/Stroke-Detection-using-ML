"""Transfer-learning CNN classifier for stroke vs. normal scans."""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import torchvision.models as tvm

    _HAS_TV = True
except ImportError:  # pragma: no cover
    _HAS_TV = False


def build_classifier(
    backbone: str = "efficientnet_b0",
    n_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Build a CNN with an ImageNet backbone and a fresh classification head."""
    if not _HAS_TV:
        raise ImportError("torchvision required for the imaging classifier")

    if backbone == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
    elif backbone == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif backbone == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'")
    return model


def target_layer(model: nn.Module, backbone: str):
    """Return the last conv layer used as the Grad-CAM target."""
    if backbone == "efficientnet_b0":
        return model.features[-1]
    if backbone in ("resnet18", "resnet50"):
        return model.layer4[-1]
    raise ValueError(f"Unknown backbone '{backbone}'")


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Softmax probability of the positive (stroke) class, index 1."""
    model.eval()
    logits = model(x.to(device))
    return torch.softmax(logits, dim=1)[:, 1]
