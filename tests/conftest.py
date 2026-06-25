# Keeps pytest from confusing the tests dir with package roots, and guards
# against the macOS dual-OpenMP-runtime crash that happens when LightGBM
# (libgomp/libomp) and PyTorch (libomp) are loaded in the same process.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Import the OpenMP-heavy libraries up front, in a stable order, before any
# test module triggers a mixed import order.
try:  # pragma: no cover - import-order guard only
    import lightgbm  # noqa: F401
    import torch  # noqa: F401
except Exception:  # pragma: no cover - libs are optional in some envs
    pass
