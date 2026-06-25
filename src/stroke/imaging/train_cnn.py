"""Training loop for the imaging classifier.

Examples
--------
Quick smoke test (synthetic data, CPU, a few steps)::

    python -m stroke.imaging.train_cnn --quick

Full training on the downloaded Kaggle CT dataset::

    python -m stroke.imaging.train_cnn \
        --data-dir data/imaging/ct --epochs 25 --batch-size 32 \
        --backbone efficientnet_b0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import config
from . import dataset as ds_mod
from . import model as model_mod


def _device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    out = {"n": int(len(labels))}
    if len(np.unique(labels)) > 1:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["pr_auc"] = float(average_precision_score(labels, probs))
    return out


def train(
    data_dir: str | None,
    backbone: str = "efficientnet_b0",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    img_size: int = 224,
    quick: bool = False,
    device: str | None = None,
    out_path: Path | None = None,
) -> dict:
    dev = _device(device)
    print(f"[imaging] device={dev} backbone={backbone} quick={quick}")

    synthetic = quick or data_dir is None
    train_loader, val_loader, n_classes, pos_index = ds_mod.build_dataloaders(
        data_dir,
        img_size=img_size,
        batch_size=batch_size,
        synthetic=synthetic,
        synthetic_n=120 if quick else 200,
    )

    model = model_mod.build_classifier(
        backbone, n_classes=n_classes, pretrained=not quick
    ).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if quick:
        epochs = 1

    best = {"pr_auc": -1.0}
    out_path = out_path or (config.IMAGING_MODELS_DIR / "cnn_classifier.pt")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if quick and step >= 3:
                break
        metrics = evaluate(model, val_loader, dev)
        print(f"[epoch {epoch + 1}/{epochs}] loss={running:.3f} val={metrics}")
        if metrics.get("pr_auc", 0) >= best["pr_auc"]:
            best = metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": backbone,
                    "img_size": img_size,
                    "pos_index": pos_index,
                    "metrics": metrics,
                },
                out_path,
            )

    print(f"[imaging] best={best} saved->{out_path}")
    with open(config.REPORTS_DIR / "imaging_metrics.json", "w") as fh:
        json.dump(best, fh, indent=2)
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the imaging stroke classifier")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--backbone", default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet18", "resnet50"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--quick", action="store_true", help="synthetic smoke test")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    train(
        data_dir=args.data_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        quick=args.quick,
        device=args.device,
    )


if __name__ == "__main__":
    main()
