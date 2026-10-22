"""Component 2 (Noise-Resilient Road Damage Segmentation Engine) routes.

Backed by PostgreSQL run and result tables. Endpoints are defined in
ml/c2/api/router.py. Mounted under /api/v1 -> /api/v1/c2/...
"""

import os
from pathlib import Path

from backend.app.repositories.c2_runs import PostgresRunStore
from ml.c2.api import build_router

REPO_DIR = Path(__file__).resolve().parents[3]
STORAGE_DIR = Path(os.environ.get("C2_STORAGE_DIR") or REPO_DIR / "ml" / "c2" / "runs")
CGAN_WEIGHTS_PATH = Path(
    os.environ.get("C2_CGAN_WEIGHTS_PATH") or REPO_DIR / "ml" / "c2" / "weights" / "cgan_frozen_sensor.pth"
)
SEGMENTER_WEIGHTS_PATH = Path(
    os.environ.get("C2_SEGMENTER_WEIGHTS_PATH") or REPO_DIR / "ml" / "c2" / "weights" / "c2_ltca_segmenter_best.pth"
)

router = build_router(
    storage_dir=STORAGE_DIR,
    cgan_weights_path=CGAN_WEIGHTS_PATH,
    segmenter_weights_path=SEGMENTER_WEIGHTS_PATH,
    store=PostgresRunStore(STORAGE_DIR),
)
