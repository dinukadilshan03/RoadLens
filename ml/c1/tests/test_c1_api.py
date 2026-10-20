"""API and adapter tests for the C1 road damage detection router."""
import io
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ml.c1.adapters.detector import CLASS_NAMES, DEFAULT_WEIGHTS_PATH, C1Detector
from ml.c1.api.router import build_router


def _make_road_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Create a synthetic asphalt road image for deterministic testing."""
    img = Image.new("RGB", (width, height), color=(72, 74, 78))
    draw = ImageDraw.Draw(img)
    # Draw simulated pothole depression and crack lines
    draw.ellipse([220, 240, 380, 360], fill=(32, 33, 36), outline=(18, 18, 20), width=4)
    draw.line([(100, 120), (140, 260), (125, 410)], fill=(22, 22, 24), width=5)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture
def dummy_weights(tmp_path) -> Path:
    w = tmp_path / "dummy_b0.pt"
    w.write_bytes(b"fake-yolo-weights-for-unit-test")
    return w


def test_model_info_endpoint(tmp_path, dummy_weights):
    app = FastAPI()
    app.include_router(
        build_router(storage_dir=tmp_path / "runs", weights_path=dummy_weights),
        prefix="/api/v1",
    )
    client = TestClient(app)

    res = client.get("/api/v1/c1/model")
    assert res.status_code == 200
    body = res.json()
    assert body["component"] == "C1"
    assert body["model_version"] == "YOLOv8m-RDD2022-B0"
    assert body["weights_ready"] is True
    assert len(body["classes"]) == 4
    assert {c["name"] for c in body["classes"]} == set(CLASS_NAMES.values())


def test_model_unavailable_returns_503(tmp_path):
    app = FastAPI()
    app.include_router(
        build_router(storage_dir=tmp_path / "runs", weights_path=tmp_path / "missing.pt"),
        prefix="/api/v1",
    )
    client = TestClient(app)

    res = client.get("/api/v1/c1/model")
    assert res.status_code == 503
    assert res.json()["code"] == "model_unavailable"


def test_upload_run_outputs_damage_class_bbox_and_confidence(tmp_path, dummy_weights):
    """Verify full C1 pipeline produces damage_class, bbox, confidence, annotated frame, and C2 crop."""
    app = FastAPI()
    app.include_router(
        build_router(storage_dir=tmp_path / "runs", weights_path=dummy_weights),
        prefix="/api/v1",
    )
    client = TestClient(app)

    fake_preds = [
        {
            "class_id": 3,
            "damage_class": "Pothole_D40",
            "display_name": "Pothole",
            "confidence": 0.8742,
            "bbox_xyxy": [220.4, 240.1, 380.8, 360.2],
        },
        {
            "class_id": 0,
            "damage_class": "Longitudinal_Crack_D00",
            "display_name": "Longitudinal Crack",
            "confidence": 0.6415,
            "bbox_xyxy": [98.0, 118.5, 145.2, 412.0],
        },
    ]

    with (
        patch("ml.c1.adapters.detector._get_yolo_model", return_value=object()),
        patch.object(C1Detector, "_predict_frame", return_value=fake_preds),
    ):
        files = [("files", ("road_sample.jpg", _make_road_jpeg(), "image/jpeg"))]
        res = client.post(
            "/api/v1/c1/runs",
            files=files,
            data={"conf_threshold": "0.25", "survey_id": "survey-pp1"},
        )
        assert res.status_code == 202
        run_id = res.json()["run_id"]

        meta = client.get(f"/api/v1/c1/runs/{run_id}").json()
        assert meta["status"] == "succeeded", meta
        assert meta["stage"] == "done"
        assert meta["survey_id"] == "survey-pp1"
        assert meta["summary"]["total_frames_processed"] == 1
        assert meta["summary"]["total_defects_detected"] == 2
        assert meta["summary"]["counts_by_class"]["Pothole_D40"] == 1
        assert meta["summary"]["counts_by_class"]["Longitudinal_Crack_D00"] == 1

        records_res = client.get(f"/api/v1/c1/runs/{run_id}/records")
        assert records_res.status_code == 200
        records = records_res.json()["records"]
        assert len(records) == 2

        pothole = next(r for r in records if r["damage_class"] == "Pothole_D40")
        assert pothole["class_id"] == 3
        assert pothole["display_name"] == "Pothole"
        assert pothole["confidence"] == 0.8742
        assert pothole["bbox"] == [220.4, 240.1, 380.8, 360.2]
        assert pothole["frame_dimensions"] == {"width": 640, "height": 480}
        assert pothole["location"] is None
        assert pothole["location_null_reason"] == "metadata_not_supplied"
        assert pothole["crop_transform"]["origin_xy"] == [220.4, 240.1]

        # Verify annotated frame and C2 crop artifact downloads
        ann_res = client.get(pothole["annotated_url"])
        assert ann_res.status_code == 200
        assert ann_res.headers["content-type"] == "image/jpeg"

        crop_res = client.get(pothole["crop_url"])
        assert crop_res.status_code == 200
        assert crop_res.headers["content-type"] == "image/jpeg"


def test_validation_errors(tmp_path, dummy_weights):
    app = FastAPI()
    app.include_router(
        build_router(storage_dir=tmp_path / "runs", weights_path=dummy_weights),
        prefix="/api/v1",
    )
    client = TestClient(app)

    # Invalid confidence threshold
    res = client.post(
        "/api/v1/c1/runs",
        files=[("files", ("road.jpg", _make_road_jpeg(), "image/jpeg"))],
        data={"conf_threshold": "1.5"},
    )
    assert res.status_code == 422
    assert res.json()["code"] == "invalid_input"

    # Unsupported extension
    res = client.post(
        "/api/v1/c1/runs",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert res.status_code == 422
    assert res.json()["code"] == "invalid_input"

    # Empty file
    res = client.post(
        "/api/v1/c1/runs",
        files=[("files", ("empty.jpg", b"", "image/jpeg"))],
    )
    assert res.status_code == 422


@pytest.mark.skipif(not DEFAULT_WEIGHTS_PATH.is_file(), reason="Real base_model_b0.pt weights not present")
def test_real_b0_model_inference(tmp_path):
    """Run real YOLOv8m base_model_b0.pt inference end-to-end when the weights file exists."""
    pytest.importorskip("ultralytics")
    app = FastAPI()
    app.include_router(
        build_router(storage_dir=tmp_path / "runs", weights_path=DEFAULT_WEIGHTS_PATH),
        prefix="/api/v1",
    )
    client = TestClient(app)

    # Use a real RDD2022 test image if available locally, otherwise synthetic road image
    rdd_test_dir = DEFAULT_WEIGHTS_PATH.parents[1] / "data" / "rdd_cleaned" / "test" / "images"
    sample_images = sorted(rdd_test_dir.glob("*.jpg"))[:1] if rdd_test_dir.is_dir() else []
    img_bytes = sample_images[0].read_bytes() if sample_images else _make_road_jpeg()

    res = client.post(
        "/api/v1/c1/runs",
        files=[("files", ("test_road.jpg", img_bytes, "image/jpeg"))],
        data={"conf_threshold": "0.20"},
    )
    assert res.status_code == 202
    run_id = res.json()["run_id"]
    meta = client.get(f"/api/v1/c1/runs/{run_id}").json()
    assert meta["status"] == "succeeded", meta
    assert meta["model_version"] == "YOLOv8m-RDD2022-B0"
    assert meta["summary"]["total_frames_processed"] == 1
