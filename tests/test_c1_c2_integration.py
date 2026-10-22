"""End-to-end integration tests between Component 1 (Detection) and Component 2 (Segmentation)."""
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

from ml.c1.adapters.detector import C1Detector
from ml.c1.api.router import build_router as build_c1_router
from ml.c2.api.router import build_router as build_c2_router


def _make_road_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Create a synthetic asphalt road image with a pothole and a crack."""
    img = Image.new("RGB", (width, height), color=(72, 74, 78))
    draw = ImageDraw.Draw(img)
    draw.ellipse([200, 220, 360, 340], fill=(28, 30, 34), outline=(15, 15, 18), width=4)
    draw.line([(90, 100), (130, 250), (115, 410)], fill=(20, 20, 22), width=5)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture
def c1_c2_client(tmp_path) -> tuple[TestClient, Path, Path]:
    c1_storage = tmp_path / "c1_runs"
    c2_storage = tmp_path / "c2_runs"
    dummy_c1_weights = tmp_path / "base_model_b0.pt"
    dummy_c1_weights.write_bytes(b"fake-c1-yolo-weights")

    app = FastAPI()
    app.include_router(
        build_c1_router(storage_dir=c1_storage, weights_path=dummy_c1_weights),
        prefix="/api/v1",
    )
    app.include_router(
        build_c2_router(
            storage_dir=c2_storage,
            cgan_weights_path=tmp_path / "missing_cgan.pth",
            segmenter_weights_path=tmp_path / "missing_seg.pth",
            c1_storage_dir=c1_storage,
        ),
        prefix="/api/v1",
    )
    return TestClient(app), c1_storage, c2_storage


def test_c1_to_c2_single_and_batch_handoff(c1_c2_client):
    client, _, _ = c1_c2_client

    fake_preds = [
        {
            "class_id": 3,
            "damage_class": "Pothole_D40",
            "display_name": "Pothole",
            "confidence": 0.915,
            "bbox_xyxy": [200.0, 220.0, 360.0, 340.0],
        },
        {
            "class_id": 0,
            "damage_class": "Longitudinal_Crack_D00",
            "display_name": "Longitudinal Crack",
            "confidence": 0.782,
            "bbox_xyxy": [80.0, 95.0, 150.0, 415.0],
        },
    ]

    with (
        patch("ml.c1.adapters.detector._get_yolo_model", return_value=object()),
        patch.object(C1Detector, "_predict_frame", return_value=fake_preds),
    ):
        c1_res = client.post(
            "/api/v1/c1/runs",
            files=[("files", ("road_test.jpg", _make_road_jpeg(), "image/jpeg"))],
            data={"conf_threshold": "0.25", "survey_id": "survey-c1-c2-e2e"},
        )
        assert c1_res.status_code == 202
        c1_run_id = c1_res.json()["run_id"]

    c1_meta = client.get(f"/api/v1/c1/runs/{c1_run_id}").json()
    assert c1_meta["status"] == "succeeded"
    assert c1_meta["summary"]["total_defects_detected"] == 2

    # 1. Verify C2 sees all available C1 crops with valid URLs and metadata
    crops_res = client.get(f"/api/v1/c2/c1-available-crops?c1_run_id={c1_run_id}")
    assert crops_res.status_code == 200
    crops_payload = crops_res.json()
    assert crops_payload["count"] == 2
    crops = crops_payload["crops"]

    pothole_crop = next(c for c in crops if c["damage_class"] == "Pothole_D40")
    assert pothole_crop["run_id"] == c1_run_id
    assert pothole_crop["survey_id"] == "survey-c1-c2-e2e"
    assert pothole_crop["bbox"] == [200.0, 220.0, 360.0, 340.0]
    assert pothole_crop["crop_url"].startswith(f"/api/v1/c1/runs/{c1_run_id}/artifacts/crops/")
    assert pothole_crop["latest_c2_run_id"] is None

    # Verify the crop_url serves the actual JPEG artifact from C1
    img_res = client.get(pothole_crop["crop_url"])
    assert img_res.status_code == 200
    assert img_res.headers["content-type"] == "image/jpeg"

    # 2. Trigger single-observation C2 run directly from C1 handoff (without re-uploading file)
    c2_trigger = client.post(
        "/api/v1/c2/runs",
        data={
            "c1_run_id": c1_run_id,
            "c1_observation_id": pothole_crop["record_id"],
        },
    )
    assert c2_trigger.status_code == 202
    c2_trigger_body = c2_trigger.json()
    c2_run_id = c2_trigger_body["run_id"]
    assert c2_trigger_body["source"] == "c1_handoff"
    assert c2_trigger_body["survey_id"] == "survey-c1-c2-e2e"
    assert c2_trigger_body["damage_class"] == "Pothole_D40"

    # 3. Verify C2 run completed with 4-part output suite and full C1 provenance
    c2_meta = client.get(f"/api/v1/c2/runs/{c2_run_id}").json()
    assert c2_meta["status"] == "succeeded", c2_meta
    assert c2_meta["source"] == "c1_handoff"
    assert c2_meta["c1_run_id"] == c1_run_id
    assert c2_meta["c1_observation_id"] == pothole_crop["record_id"]

    record_res = client.get(f"/api/v1/c2/runs/{c2_run_id}/record")
    assert record_res.status_code == 200
    record = record_res.json()
    assert record["upstream"]["c1_run_id"] == c1_run_id
    assert record["upstream"]["c1_observation_id"] == pothole_crop["record_id"]
    assert record["upstream"]["c1_defect_id"] == pothole_crop["defect_id"]
    assert record["upstream"]["detection_confidence"] == 0.915
    assert record["upstream"]["bbox"] == [200.0, 220.0, 360.0, 340.0]
    assert record["pixel_transform"]["origin_xy"] == [200.0, 220.0]
    assert record["pixel_transform"]["c2_canvas_size"] == [256, 256]
    assert record["pixel_transform"]["scale_x_to_c2"] == round(256.0 / 160.0, 6)
    assert record["pixel_transform"]["scale_y_to_c2"] == round(256.0 / 120.0, 6)

    for artifact_kind in ("input_rgb", "pseudo_thermal", "mask_2d", "skeleton_1px", "confidence_npz", "visual_overlay"):
        art_res = client.get(f"/api/v1/c2/runs/{c2_run_id}/artifacts/{artifact_kind}")
        assert art_res.status_code == 200, f"Artifact {artifact_kind} missing"

    # Verify latest_c2_run_id is now populated on c1-available-crops
    crops_after = client.get(f"/api/v1/c2/c1-available-crops?c1_run_id={c1_run_id}").json()["crops"]
    updated_pothole = next(c for c in crops_after if c["record_id"] == pothole_crop["record_id"])
    assert updated_pothole["latest_c2_run_id"] == c2_run_id

    # 4. Trigger batch handoff for the entire C1 run
    batch_res = client.post(
        "/api/v1/c2/runs/from-c1",
        data={"c1_run_id": c1_run_id},
    )
    assert batch_res.status_code == 202
    batch_body = batch_res.json()
    assert batch_body["c1_run_id"] == c1_run_id
    assert batch_body["count"] == 2
    assert len(batch_body["runs"]) == 2

    for r_item in batch_body["runs"]:
        r_detail = client.get(f"/api/v1/c2/runs/{r_item['run_id']}").json()
        assert r_detail["status"] == "succeeded"
        assert r_detail["source"] == "c1_handoff"


def test_c2_handoff_missing_c1_crop_returns_404(c1_c2_client):
    client, _, _ = c1_c2_client
    res = client.post(
        "/api/v1/c2/runs",
        data={
            "c1_run_id": "0" * 32,
            "c1_observation_id": "obs_nonexistent",
        },
    )
    assert res.status_code == 404
    assert res.json()["code"] == "c1_crop_not_found"
