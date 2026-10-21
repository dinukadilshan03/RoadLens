"""C1 API against a real PostgreSQL database.

Skipped unless C1_PG_TEST=1 and the POSTGRES_* variables point to a disposable test database, e.g.
    docker compose up -d --wait db
    C1_PG_TEST=1 POSTGRES_PASSWORD=... python -m pytest tests/test_c1_postgres.py -q
"""
import io
import os
from unittest.mock import patch

import pytest

if os.environ.get("C1_PG_TEST") != "1":
    pytest.skip("set C1_PG_TEST=1 to run against PostgreSQL", allow_module_level=True)

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.repositories.c1_runs import PostgresRunStore
from ml.c1.adapters.detector import C1Detector
from ml.c1.api import build_router


def _sample_jpeg() -> bytes:
    img = Image.new("RGB", (640, 480), color=(70, 70, 75))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    base = tmp_path_factory.mktemp("c1pg")
    weights = base / "base_model_b0.pt"
    weights.write_bytes(b"fake-b0-weights")
    store = PostgresRunStore(base / "runs")
    app = FastAPI()
    app.include_router(
        build_router(storage_dir=base / "runs", weights_path=weights, store=store),
        prefix="/api/v1",
    )
    return TestClient(app), store


def test_c1_run_and_observations_stored_in_postgres(client):
    c, store = client
    fake_preds = [
        {
            "class_id": 3,
            "damage_class": "Pothole_D40",
            "display_name": "Pothole",
            "confidence": 0.912,
            "bbox_xyxy": [120.0, 180.0, 260.0, 310.0],
        },
        {
            "class_id": 2,
            "damage_class": "Alligator_Crack_D20",
            "display_name": "Alligator Crack",
            "confidence": 0.735,
            "bbox_xyxy": [300.0, 200.0, 520.0, 410.0],
        },
    ]

    with (
        patch("ml.c1.adapters.detector._get_yolo_model", return_value=object()),
        patch.object(C1Detector, "_predict_frame", return_value=fake_preds),
    ):
        r = c.post(
            "/api/v1/c1/runs",
            files=[("files", ("road.jpg", _sample_jpeg(), "image/jpeg"))],
            data={"survey_id": "pg-c1-test", "conf_threshold": "0.25"},
        )
        assert r.status_code == 202
        run_id = r.json()["run_id"]

    meta = c.get(f"/api/v1/c1/runs/{run_id}").json()
    assert meta["status"] == "succeeded", meta
    assert meta["model_version"] == "YOLOv8m-RDD2022-B0"
    assert meta["summary"]["total_defects_detected"] == 2

    recs = c.get(f"/api/v1/c1/runs/{run_id}/records").json()["records"]
    assert len(recs) == 2

    with store._conn() as conn:
        n = conn.execute("SELECT count(*) AS n FROM c1_observations WHERE run_id = %s", (run_id,)).fetchone()["n"]
        rows = conn.execute(
            "SELECT damage_class, confidence, bbox FROM c1_observations WHERE run_id = %s ORDER BY id",
            (run_id,),
        ).fetchall()
    assert n == 2
    assert rows[0]["damage_class"] == "Pothole_D40"
    assert rows[0]["confidence"] == 0.912
    assert rows[0]["bbox"] == [120.0, 180.0, 260.0, 310.0]
    assert rows[1]["damage_class"] == "Alligator_Crack_D20"
