"""Component 1 (road damage detection) routes, backed by PostgreSQL run/observation tables.

Endpoints are defined in ml/c1/api/router.py (owned with the C1 adapter); this module supplies the
platform persistence. Mounted under /api/v1 -> /api/v1/c1/...
"""
import os
from pathlib import Path

from backend.app.repositories.c1_runs import PostgresRunStore
from ml.c1.api import build_router

REPO_DIR = Path(__file__).resolve().parents[3]
STORAGE_DIR = Path(os.environ.get("C1_STORAGE_DIR") or REPO_DIR / "ml" / "c1" / "runs")
WEIGHTS_PATH = Path(os.environ.get("C1_WEIGHTS_PATH") or REPO_DIR / "ml" / "c1" / "weights" / "base_model_b0.pt")

router = build_router(
    storage_dir=STORAGE_DIR,
    weights_path=WEIGHTS_PATH,
    store=PostgresRunStore(STORAGE_DIR),
)
