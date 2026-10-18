"""Component 4 (road roughness) routes, backed by PostgreSQL run/segment tables.

Endpoints are defined in ml/c4/api/router.py (owned with the C4 adapter); this module supplies the
platform persistence. Mounted under /api/v1 -> /api/v1/c4/...
"""
import os
from pathlib import Path

from backend.app.repositories.c4_runs import PostgresRunStore
from ml.c4.api import build_router

REPO_DIR = Path(__file__).resolve().parents[3]
STORAGE_DIR = Path(os.environ.get("C4_STORAGE_DIR") or REPO_DIR / "ml" / "c4" / "runs")

router = build_router(store=PostgresRunStore(STORAGE_DIR))
