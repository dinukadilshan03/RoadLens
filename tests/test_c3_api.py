"""Exercise the real C3 checkpoint through the public upload and handoff contract."""

import io
import json
import numpy as np
import pytest
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.app.routes.c3 import router
from backend.app.services.c3 import lifespan, surface_analysis


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture(scope="module")
def client():
    app = FastAPI(lifespan=lifespan)
    app.include_router(router, prefix="/api/v1")
    with TestClient(app) as client:
        assert client.get("/api/v1/c3/model").json()["ready"]
        yield client


def inputs():
    rgb = np.full((96, 128, 3), 100, dtype=np.uint8)
    mask = np.zeros((96, 128), dtype=np.uint8)
    mask[30:65, 40:85] = 255
    return rgb, mask


def test_manual_real_inference(client):
    rgb, mask = inputs()
    response = client.post(
        "/api/v1/c3/analyze",
        files={
            "image": ("road.png", png(rgb)),
            "mask": ("mask.png", png(mask)),
            "crack_mask": ("crack.png", png(mask)),
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    defect = result["defects"][0]
    assert defect["source_bbox_xyxy"] == [40, 30, 85, 65]
    assert defect["pixel_measurements"]["footprint_area_px2"] == 35 * 45
    assert result["defects"][1]["damage_type"] == "crack"
    assert result["defects"][1]["pixel_measurements"]["crack"] is not None
    assert np.asarray(defect["raw_axial_depression_map"]).shape == (192, 256)
    assert defect["geometry_raw"]["max_raw"] > 0
    assert defect["severity"]["physical_level"] is None
    assert defect["physical_measurements"]["depth_d95_mm"] is None
    assert result["map_coordinates"]["space"] == "full_frame"


@pytest.mark.parametrize("kind", ["empty", "corrupt"])
def test_invalid_masks(client, kind):
    rgb, mask = inputs()
    if kind == "empty":
        mask[:] = 0
    data = b"not an image" if kind == "corrupt" else png(mask)
    response = client.post(
        "/api/v1/c3/analyze",
        files={"image": ("road.png", png(rgb)), "mask": ("mask.png", data)},
    )
    assert response.status_code == 422


def test_shared_automatic_handoff(client):
    rgb, mask = inputs()
    records = [
        {
            "defect_id": "c1-c2-1",
            "bbox_xyxy": [40, 30, 85, 65],
            "mask_space": "frame",
            "mask": mask.tolist(),
            "score": 0.9,
        }
    ]
    response = client.post(
        "/api/v1/c3/analyze",
        files={"image": ("road.png", png(rgb))},
        data={"detections": json.dumps(records)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["defects"][0]["defect_id"] == "c1-c2-1"


def test_unavailable_model(client, monkeypatch):
    rgb, mask = inputs()
    monkeypatch.setattr(surface_analysis, "bridge", None)
    response = client.post(
        "/api/v1/c3/analyze",
        files={"image": ("road.png", png(rgb)), "mask": ("mask.png", png(mask))},
    )
    assert response.status_code == 503


@pytest.mark.parametrize("format", ["JPEG", "WEBP", "BMP", "PNG"])
def test_flexible_uploads(client, format):
    rgb, mask = inputs()
    small = np.repeat(mask[::2, ::2, None], 3, axis=2)
    small[0, 0] = 30

    def encode(a):
        out = io.BytesIO()
        Image.fromarray(a).save(out, format=format)
        return out.getvalue()

    response = client.post(
        "/api/v1/c3/analyze",
        files={
            "image": ("road", encode(rgb)),
            "mask": ("mask", encode(small)),
            "crack_mask": ("crack", encode(small)),
        },
    )
    assert response.status_code == 422, response.text


def test_crack_only_without_model(client, monkeypatch):
    rgb, mask = inputs()
    monkeypatch.setattr(surface_analysis, "bridge", None)
    response = client.post(
        "/api/v1/c3/analyze",
        files={"image": ("road.png", png(rgb)), "crack_mask": ("crack.png", png(mask))},
    )
    assert response.status_code == 200, response.text
    defect = response.json()["defects"][0]
    assert defect["damage_type"] == "crack"
    assert defect["geometry_raw"] is None
    assert defect["presentation"]["severity"] in {"Low", "High", "Needs Review"}
    assert defect["presentation"]["physical_severity"] == "Not established"
    assert defect["visuals"]["centerline"].startswith("data:image/png;base64,")


def test_multiple_defects(client):
    rgb, mask = inputs()
    mask[5:15, 5:15] = 255
    response = client.post(
        "/api/v1/c3/analyze",
        files={
            "image": ("road.png", png(rgb)),
            "mask": ("mask.png", png(mask)),
            "crack_mask": ("crack.png", png(mask)),
        },
    )
    assert response.status_code == 200, response.text
    defects = response.json()["defects"]
    assert [d["presentation"]["label"] for d in defects] == [
        "Pothole 1",
        "Pothole 2",
        "Crack Network",
    ]
    assert len({d["defect_id"] for d in defects}) == 3
    assert defects[-1]["crack_network"]["connected_regions"] == 2


@pytest.mark.parametrize("kind", ["missing", "photo", "colored", "misaligned", "full"])
def test_reject_ambiguous_inputs(client, kind):
    rgb, mask = inputs()
    data = {"image": ("road.png", png(rgb))}
    if kind != "missing":
        if kind == "photo":
            mask = rgb
        if kind == "colored":
            mask = np.stack([mask, np.zeros_like(mask), mask], axis=-1)
        if kind == "misaligned":
            mask = mask[::2, ::2]
        if kind == "full":
            mask[:] = 255
        data["mask"] = ("mask.png", png(mask))
    response = client.post("/api/v1/c3/analyze", files=data)
    assert response.status_code == 422, response.text


def test_field_severity_is_separate_from_model(client):
    rgb, mask = inputs()
    field = {
        "pothole": {
            "source": "MANUAL_FIELD_MEASUREMENT",
            "road_surface_type": "ASPHALT_CONCRETE",
            "fhwa_150mm_inscribed_circle_confirmed": True,
            "healthy_road_reference_verified": True,
            "measurement_method": "STRAIGHTEDGE_RULER",
            "deepest_point_mm": 42,
            "depth_uncertainty_mm": 1,
        }
    }
    response = client.post(
        "/api/v1/c3/analyze",
        files={"image": ("road.png", png(rgb)), "mask": ("mask.png", png(mask))},
        data={"field_measurements": json.dumps(field)},
    )
    assert response.status_code == 200, response.text
    defect = response.json()["defects"][0]
    assert defect["presentation"]["physical_severity"] == "Moderate"
    visual_level = defect["presentation"]["severity"]
    assert defect["severity"]["physical_level"] is None
    assert "42 mm" in defect["presentation"]["measurement"]
    field["pothole"]["deepest_point_mm"] = 25
    response = client.post(
        "/api/v1/c3/analyze",
        files={"image": ("road.png", png(rgb)), "mask": ("mask.png", png(mask))},
        data={"field_measurements": json.dumps(field)},
    )
    assert (
        response.json()["defects"][0]["presentation"]["physical_severity"]
        == "Not established"
    )
    assert response.json()["defects"][0]["presentation"]["severity"] == visual_level


def test_pdf_report_contains_each_defect(client):
    rgb, mask = inputs()
    mask[5:15, 5:15] = 255
    response = client.post(
        "/api/v1/c3/report",
        files={
            "image": ("road.png", png(rgb)),
            "mask": ("mask.png", png(mask)),
            "crack_mask": ("crack.png", png(mask)),
        },
    )
    assert response.status_code == 200, response.text[:200]
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    import re

    assert len(re.findall(rb"/Type /Page\b", response.content)) == 6


def test_automatic_crack_roi(client):
    rgb, mask = inputs()
    records = [
        {
            "defect_id": "c2-crack",
            "damage_type": "crack",
            "bbox_xyxy": [40, 30, 85, 65],
            "mask_space": "bbox",
            "mask": mask[30:65, 40:85].tolist(),
        }
    ]
    response = client.post(
        "/api/v1/c3/analyze",
        files={"image": ("road.png", png(rgb))},
        data={"detections": json.dumps(records)},
    )
    assert response.status_code == 200, response.text
    defect = response.json()["defects"][0]
    assert defect["damage_type"] == "crack"
    assert defect["source_bbox_xyxy"] == [40, 30, 85, 65]
    assert defect["presentation"]["assessment"] in {"Analysis completed", "Needs review"}


def test_actual_visual_models_are_independent(client):
    rgb, mask = inputs()
    response = client.post("/api/v1/c3/analyze", files={
        "image": ("road.png", png(rgb)), "mask": ("mask.png", png(mask)),
        "crack_mask": ("crack.png", png(mask)),
    })
    pothole, crack = response.json()["defects"]
    assert pothole["visual_severity"]["model_version"] == "C3-22.0.0-experimental"
    assert crack["visual_severity"]["model_version"] == "C3-27.0.0-experimental"
    assert crack["visual_severity"]["analysis_mode"] == "RGB_ONLY"
    assert crack["visual_severity"]["suggested_visual_class"] in {"LOW", "HIGH"}
    assert pothole["visual_severity"]["suggested_class"] in {"LOW", "MODERATE", "HIGH"}
    assert all(d["visual_severity"]["physical_severity"] is None for d in (pothole, crack))


def test_visual_failure_retains_geometry(client, monkeypatch):
    rgb, mask = inputs()
    monkeypatch.setattr(surface_analysis, "visual_models", {})
    response = client.post("/api/v1/c3/analyze", files={
        "image": ("road.png", png(rgb)), "mask": ("mask.png", png(mask)),
    })
    assert response.status_code == 200
    defect = response.json()["defects"][0]
    assert defect["relative_surface"]
    assert defect["visual_severity"] is None
    assert defect["presentation"]["severity"] == "Needs Review"
    assert defect["presentation"]["reliability"] == "Needs review"


def test_border_input_overrides_visual_category(client):
    rgb, mask = inputs()
    mask[:40, 40:85] = 255
    response = client.post("/api/v1/c3/analyze", files={
        "image": ("road.png", png(rgb)), "crack_mask": ("crack.png", png(mask)),
    })
    defect = response.json()["defects"][0]
    assert defect["presentation"]["severity"] == "Needs Review"
    assert defect["presentation"]["review_reasons"]


def test_research_evidence_is_allowlisted(client):
    assert client.get("/api/v1/c3/research/gate").json()["learned_reference_deployment_authorized"] is False
    assert client.get("/api/v1/c3/research/stages").headers["content-type"] == "image/png"
    assert client.get("/api/v1/c3/research/selected_model.pt").status_code == 404


def test_isolated_crack_pixel_is_insufficient_input(client):
    rgb, mask = inputs()
    mask[:] = 0
    mask[40, 50] = 255
    response = client.post("/api/v1/c3/analyze", files={
        "image": ("road.png", png(rgb)), "crack_mask": ("crack.png", png(mask)),
    })
    assert response.status_code == 200
    defect = response.json()["defects"][0]
    assert defect["presentation"]["reliability"] == "Insufficient input"
    assert defect["presentation"]["severity"] == "Needs Review"


def test_wrong_crack_mask_does_not_block_valid_pothole(client):
    rgb, mask = inputs()
    response = client.post("/api/v1/c3/analyze", files={
        "image": ("road.png", png(rgb)), "mask": ("pothole.png", png(mask)),
        "crack_mask": ("other-photo.png", png(mask[::2, ::2])),
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert [d["damage_type"] for d in result["defects"]] == ["pothole"]
    warning = result["input_warnings"][0]
    assert warning["damage_type"] == "crack"
    assert "64 x 48" in warning["message"]
    assert "128 x 96" in warning["message"]
    assert "not analyzed" in warning["message"]


def test_wrong_crack_mask_alone_explains_how_to_fix(client):
    rgb, mask = inputs()
    response = client.post("/api/v1/c3/analyze", files={
        "image": ("road.png", png(rgb)),
        "crack_mask": ("other-photo.png", png(mask[::2, ::2])),
    })
    assert response.status_code == 422
    assert "Crack mask" in response.json()["detail"]
    assert "same photo" in response.json()["detail"]


def test_pdf_available_when_one_mask_is_invalid(client):
    rgb, mask = inputs()
    response = client.post("/api/v1/c3/report", files={
        "image": ("road.png", png(rgb)), "mask": ("pothole.png", png(mask)),
        "crack_mask": ("other-photo.png", png(mask[::2, ::2])),
    })
    assert response.status_code == 200, response.text[:200]
    assert response.content.startswith(b"%PDF")


def independent_inputs():
    rgb, mask = inputs()
    crack = np.zeros((384, 544), dtype=np.uint8)
    crack[100:280, 240:244] = 255
    return rgb, mask, crack


def test_independent_crack_mask_has_own_coordinates(client):
    rgb, mask, crack = independent_inputs()
    response = client.post("/api/v1/c3/assess", files={
        "image": ("pothole-photo.png", png(rgb)), "mask": ("pothole-mask.png", png(mask)),
        "crack_mask": ("separate-crack.png", png(crack)),
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["input_warnings"] == []
    pothole, crack_result = result["defects"]
    assert (pothole["source_width"], pothole["source_height"]) == (128, 96)
    assert (crack_result["source_width"], crack_result["source_height"]) == (544, 384)
    assert crack_result["source_bbox_xyxy"] == [240, 100, 244, 280]
    assert crack_result["source_group"] == "crack"
    assert crack_result["source_kind"] == "mask"
    assert crack_result["visual_severity"] is None
    assert crack_result["pixel_measurements"]["crack"]["area_px2"] == 180 * 4
    assert crack_result["presentation"]["reliability"] == "Shape analyzed"


def test_mask_only_crack_never_runs_rgb_classifier(client, monkeypatch):
    _, _, crack = independent_inputs()
    class NoRGBPrediction:
        def predict(self, *args):
            pytest.fail("Mask-only input must never reach the RGB classifier")
    monkeypatch.setitem(surface_analysis.visual_models, "crack", NoRGBPrediction())
    response = client.post("/api/v1/c3/assess", files={"crack_mask": ("crack.png", png(crack))})
    assert response.status_code == 200, response.text
    defect = response.json()["defects"][0]
    assert defect["analysis_mode"] == "MASK_SHAPE_ONLY"
    assert defect["presentation"]["severity"] == "Add crack photo"


def test_crack_classifier_receives_own_photo(client, monkeypatch):
    rgb, mask, crack = independent_inputs()
    own_rgb = np.full((*crack.shape, 3), 173, dtype=np.uint8)
    observed = []
    class OwnPhotoPrediction:
        def predict(self, image, bbox):
            observed.append((image.shape, int(image[0, 0, 0]), bbox))
            return {"visual_crack_severity": "LOW", "review_required": False, "physical_severity": None}
    monkeypatch.setitem(surface_analysis.visual_models, "crack", OwnPhotoPrediction())
    response = client.post("/api/v1/c3/assess", files={
        "image": ("pothole.png", png(rgb)), "mask": ("pothole-mask.png", png(mask)),
        "crack_mask": ("crack-mask.png", png(crack)), "crack_image": ("crack-photo.png", png(own_rgb)),
    })
    assert response.status_code == 200, response.text
    assert observed == [((384, 544, 3), 173, [240, 100, 244, 280])]
    assert response.json()["defects"][1]["presentation"]["severity"] == "Low"


def test_independent_sources_pdf(client):
    rgb, mask, crack = independent_inputs()
    response = client.post("/api/v1/c3/assessment-report", files={
        "image": ("pothole.png", png(rgb)), "mask": ("pothole-mask.png", png(mask)),
        "crack_mask": ("crack.png", png(crack)),
    })
    assert response.status_code == 200, response.text[:200]
    import re
    assert len(re.findall(rb"/Type /Page\b", response.content)) == 5


def test_optional_crack_photo_must_match_its_own_mask(client):
    rgb, mask, crack = independent_inputs()
    response = client.post("/api/v1/c3/assess", files={
        "image": ("pothole.png", png(rgb)), "mask": ("pothole-mask.png", png(mask)),
        "crack_mask": ("crack.png", png(crack)), "crack_image": ("wrong-crack-photo.png", png(rgb)),
    })
    assert response.status_code == 200
    assert len(response.json()["defects"]) == 1
    assert response.json()["input_warnings"][0]["damage_type"] == "crack"


def test_independent_crack_runs_when_pothole_model_is_unavailable(client, monkeypatch):
    rgb, mask, crack = independent_inputs()
    monkeypatch.setattr(surface_analysis, "bridge", None)
    response = client.post("/api/v1/c3/assess", files={
        "image": ("pothole.png", png(rgb)), "mask": ("pothole-mask.png", png(mask)),
        "crack_mask": ("crack.png", png(crack)),
    })
    assert response.status_code == 200
    assert [d["damage_type"] for d in response.json()["defects"]] == ["crack"]
    assert response.json()["input_warnings"][0]["damage_type"] == "pothole"


def test_misaligned_automatic_bbox(client):
    rgb, mask = inputs()
    records = [
        {
            "defect_id": "c2",
            "bbox_xyxy": [0, 0, 10, 10],
            "mask_space": "frame",
            "mask": mask.tolist(),
        }
    ]
    response = client.post(
        "/api/v1/c3/analyze",
        files={"image": ("road.png", png(rgb))},
        data={"detections": json.dumps(records)},
    )
    assert response.status_code == 422


def test_shared_photo_crack_network_not_aggregated(client, monkeypatch):
    rgb, mask = inputs()
    mask[:] = 0
    mask[20:70, 35:38] = 255
    mask[45:48, 20:90] = 255
    mask[15:35, 105:108] = 255
    seen = []
    class RegionalModel:
        def predict(self, photo, bbox):
            seen.append(int(photo[0, 0, 0]))
            return {"visual_crack_severity": "HIGH", "review_required": False}
    monkeypatch.setitem(surface_analysis.visual_models, "crack", RegionalModel())
    response = client.post("/api/v1/c3/assess", files={
        "image": ("road.png", png(rgb)), "crack_mask": ("crack.png", png(mask)),
    })
    assert response.status_code == 200, response.text
    assert len(response.json()["defects"]) == 1
    defect = response.json()["defects"][0]
    network = defect["crack_network"]
    assert network["connected_regions"] == 2
    assert network["branched"]
    assert len(network["regional_visual_findings"]) == 2
    assert seen == [100, 100]
    assert defect["source_kind"] == "photo"
    assert defect["presentation"]["severity"] == "Needs Review"
    assert network["aggregation_validated"] is False
    assert network["original_measurements"]["area_px2"] == int((mask > 0).sum())
    assert len(network["branches"]) > 1
    import base64
    for region in network["regions"]:
        x0, y0, x1, y1 = region["bbox_xyxy"]
        overlay = np.array(Image.open(io.BytesIO(base64.b64decode(region["overlay"].split(",", 1)[1]))))
        assert overlay.shape == (y1-y0, x1-x0, 4)
        selected_pixels = overlay[..., 3] > 0
        assert int(selected_pixels.sum()) == region["pixels"]
        assert (overlay[selected_pixels, :3] == [230, 35, 45]).all()
        assert (mask[y0:y1, x0:x1][selected_pixels] > 0).all()



def test_explicit_instances_are_preserved():
    from backend.app.services.c3 import normalize_records
    rgb, mask = inputs()
    records = [{"defect_id": name, "damage_type": "crack", "mask_space": "frame",
                "bbox_xyxy": [0, 0, 128, 96], "mask": mask} for name in ("instance-a", "instance-b")]
    assert [r["defect_id"] for r in normalize_records(rgb, records)] == ["instance-a", "instance-b"]
