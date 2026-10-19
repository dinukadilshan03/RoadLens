"""Component 1 (Road Damage Detection & Mapping) route mount.

Mounted under /api/v1 -> /api/v1/c1/...
"""

from __future__ import annotations

import os
from pathlib import Path

from ml.c1.api.router import build_c1_router
from ml.c1.api.store import C1RunStore

REPO_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = Path(os.environ.get("C1_STORAGE_DIR") or REPO_DIR / "ml" / "c1" / "runs")
WEIGHTS_PATH = Path(os.environ.get("C1_WEIGHTS_PATH") or REPO_DIR / "ml" / "c1" / "weights" / "base_model_b0.pt")

router = build_c1_router(
    storage_dir=STORAGE_DIR,
    weights_path=WEIGHTS_PATH,
    store=C1RunStore(STORAGE_DIR),
)
