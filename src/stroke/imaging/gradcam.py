"""Grad-CAM saliency for the imaging classifier.

Uses the ``grad-cam`` package when available, with a compact built-in
fallback so heatmaps work with only torch installed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class _SimpleGradCAM:
    """Minimal Grad-CAM: hooks one conv layer, weights maps by gradients."""

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, output):
        self.activations = output.detach()

    def _bwd(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        logits = self.model(x)
        self.model.zero_grad()
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def compute_gradcam(model, target_layer, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
    """Return a normalised [0,1] heatmap (H, W) for a single image tensor."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    try:
        from pytorch_grad_cam import GradCAM

        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale = cam(input_tensor=x)[0]
        return grayscale
    except Exception:
        return _SimpleGradCAM(model, target_layer)(x, class_idx)


def overlay_heatmap(image_hw: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a heatmap over a grayscale image -> RGB uint8 array."""
    try:
        import cv2

        heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    except ImportError:
        import matplotlib.cm as cm

        heat = cm.jet(cam)[..., :3]
    base = np.stack([image_hw] * 3, axis=-1)
    base = (base - base.min()) / (base.ptp() + 1e-8)
    blended = (1 - alpha) * base + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype("uint8")
