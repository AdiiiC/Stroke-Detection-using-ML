"""Imaging branch smoke tests. Skipped entirely if torch is absent."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stroke.imaging import fusion
from stroke.imaging.dataset import SyntheticCTDataset, build_dataloaders


def test_synthetic_dataset_shapes():
    ds = SyntheticCTDataset(n=10, img_size=64)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y in (0, 1)


def test_build_dataloaders_synthetic():
    tr, va, n_classes, pos = build_dataloaders(
        None, img_size=64, batch_size=4, synthetic=True, synthetic_n=20
    )
    xb, yb = next(iter(tr))
    assert xb.shape[1:] == (3, 64, 64)
    assert n_classes == 2


def test_classifier_forward_and_gradcam():
    pytest.importorskip("torchvision")
    from stroke.imaging.gradcam import compute_gradcam
    from stroke.imaging.model import build_classifier, target_layer

    model = build_classifier("resnet18", n_classes=2, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 2)
    cam = compute_gradcam(model, target_layer(model, "resnet18"), x, class_idx=1)
    assert cam.ndim == 2
    assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0


def test_unet_forward_and_losses():
    from stroke.imaging.unet import UNet, bce_dice_loss, dice_score

    net = UNet(in_channels=1, n_classes=1, base=8)
    x = torch.randn(2, 1, 64, 64)
    logits = net(x)
    assert logits.shape == (2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss = bce_dice_loss(logits, target)
    assert loss.item() > 0
    assert 0.0 <= dice_score(logits, target) <= 1.0


def test_weighted_fusion_picks_best_weight():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=200) < 0.3).astype(int)
    # tabular informative, imaging noise -> weight should lean tabular.
    p_tab = np.clip(y * 0.6 + rng.uniform(0, 0.4, 200), 0, 1)
    p_img = rng.uniform(0, 1, 200)
    fused = fusion.WeightedFusion.fit(p_tab, p_img, y)
    assert 0.0 <= fused.weight <= 1.0
    out = fused.predict_proba(p_tab, p_img)
    assert out.shape == (200,)


def test_stacked_fusion_runs():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=150) < 0.4).astype(int)
    p_tab = np.clip(y * 0.5 + rng.uniform(0, 0.5, 150), 0, 1)
    p_img = np.clip(y * 0.5 + rng.uniform(0, 0.5, 150), 0, 1)
    model = fusion.StackedFusion().fit(p_tab, p_img, y)
    out = model.predict_proba(p_tab, p_img)
    assert ((out >= 0) & (out <= 1)).all()
