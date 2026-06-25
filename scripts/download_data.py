"""Download brain-stroke imaging datasets via the Kaggle API.

Prerequisites
-------------
* ``pip install -r requirements-imaging.txt``
* A configured ``~/.kaggle/kaggle.json`` (API token) with permission to
  the datasets below. Accept each dataset's rules on its Kaggle page first.

Datasets
--------
* ``ct``  -> Brain Stroke CT Image Dataset (classification: normal vs stroke)
            slug: ``afridirahman/brain-stroke-ct-image-dataset``
* You can register additional sources (e.g. ISLES/AISD for MRI lesion
  segmentation) by extending ``DATASETS`` below; those typically require
  manual download due to licensing.

Usage
-----
    python scripts/download_data.py --dataset ct
    python scripts/download_data.py --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGING_DIR = ROOT / "data" / "imaging"

# name -> (kaggle slug, is_competition)
DATASETS: dict[str, tuple[str, bool]] = {
    "ct": ("afridirahman/brain-stroke-ct-image-dataset", False),
}


def _ensure_kaggle():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        sys.exit(
            "kaggle package not installed. Run:\n"
            "  pip install -r requirements-imaging.txt"
        )
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        sys.exit(
            f"Kaggle credentials not found at {cred}.\n"
            "Create an API token at https://www.kaggle.com/settings and place "
            "kaggle.json there (chmod 600)."
        )


def download(name: str):
    if name not in DATASETS:
        sys.exit(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    slug, is_comp = DATASETS[name]
    out = IMAGING_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle"]
    cmd += (
        ["competitions", "download", "-c", slug]
        if is_comp
        else ["datasets", "download", "-d", slug]
    )
    cmd += ["-p", str(out), "--unzip"]
    print(f"[download] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[done] extracted to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Download imaging datasets")
    p.add_argument("--dataset", choices=list(DATASETS), help="single dataset key")
    p.add_argument("--all", action="store_true", help="download every dataset")
    args = p.parse_args(argv)

    _ensure_kaggle()
    if args.all:
        for name in DATASETS:
            download(name)
    elif args.dataset:
        download(args.dataset)
    else:
        p.error("pass --dataset <key> or --all")


if __name__ == "__main__":
    main()
