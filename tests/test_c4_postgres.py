"""C4 API against a real PostgreSQL database.

Skipped unless C4_PG_TEST=1 and the POSTGRES_* variables point to a disposable test database, e.g.
    docker compose up -d --wait db
    C4_PG_TEST=1 POSTGRES_PASSWORD=... python -m pytest tests/test_c4_postgres.py -q
"""
import os
from pathlib import Path

import pytest

if os.environ.get("C4_PG_TEST") != "1":
    pytest.skip("set C4_PG_TEST=1 to run against PostgreSQL", allow_module_level=True)
pytest.importorskip("xgboost")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.repositories.c4_runs import PostgresRunStore
from ml.c4.api import build_router

FIXTURE = Path(__file__).resolve().parents[1] / "ml" / "c4" / "fixtures" / "synthetic_drive"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    base = tmp_path_factory.mktemp("c4pg")
    store = PostgresRunStore(base / "runs")
    app = FastAPI()
    app.include_router(build_router(store=store, model_dir=base / "none", allow_demo=True,
                                    demo_dir=base / "demo"), prefix="/api/v1")
    return TestClient(app), store


def test_upload_run_is_stored_in_postgres(client):
    c, store = client
    files = {name: (f"{name}.csv", (FIXTURE / f"{name}.csv").read_bytes(), "text/csv") for name in ("imu", "gps")}
    files["manifest"] = ("manifest.json", (FIXTURE / "manifest.json").read_bytes(), "application/json")
    r = c.post("/api/v1/c4/runs", files=files, data={"survey_id": "pg-test"})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    meta = c.get(f"/api/v1/c4/runs/{run_id}").json()
    assert meta["status"] == "succeeded" and meta["model_kind"] == "demo"
    assert meta["model_version"] == "demo-1"

    recs = c.get(f"/api/v1/c4/runs/{run_id}/records").json()["records"]
    gj = c.get(f"/api/v1/c4/runs/{run_id}/segments").json()
    assert len(recs) == len(gj["features"]) == meta["summary"]["segments"]["count"] > 0
    assert [f["properties"]["interval_start_ms"] for f in gj["features"]] == sorted(
        f["properties"]["interval_start_ms"] for f in gj["features"])

    with store._conn() as conn:
        n = conn.execute("SELECT count(*) AS n FROM c4_segments WHERE run_id = %s", (run_id,)).fetchone()["n"]
        classes = conn.execute("SELECT DISTINCT roughness_class FROM c4_segments WHERE run_id = %s",
                               (run_id,)).fetchall()
    assert n == len(recs)
    assert {r["roughness_class"] for r in classes} <= {"good", "regular", "bad"}
    assert any(x["run_id"] == run_id for x in c.get("/api/v1/c4/runs").json()["runs"])


def test_failed_run_is_recorded(client):
    c, _ = client
    bad = (FIXTURE / "manifest.json").read_text().replace('"gravity_included": true', '"gravity_included": false')
    files = {name: (f"{name}.csv", (FIXTURE / f"{name}.csv").read_bytes(), "text/csv") for name in ("imu", "gps")}
    r = c.post("/api/v1/c4/runs", files=files, data={"manifest_json": bad})
    meta = c.get(f"/api/v1/c4/runs/{r.json()['run_id']}").json()
    assert meta["status"] == "failed" and meta["error"]["code"] == "invalid_input"
    assert c.get(f"/api/v1/c4/runs/{meta['run_id']}/segments").status_code == 409
