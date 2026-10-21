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
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


@pytest.fixture(scope='module')
def client():
    app = FastAPI(lifespan=lifespan)
    app.include_router(router, prefix='/api/v1')
    with TestClient(app) as client:
        assert client.get('/api/v1/c3/model').json()['ready']
        yield client


def inputs():
    rgb = np.full((96, 128, 3), 100, dtype=np.uint8)
    mask = np.zeros((96, 128), dtype=np.uint8)
    mask[30:65, 40:85] = 255
    return rgb, mask


def test_manual_real_inference(client):
    rgb, mask = inputs()
    response = client.post('/api/v1/c3/analyze', files={'image': ('road.png', png(rgb)), 'mask': ('mask.png', png(mask)), 'crack_mask': ('crack.png', png(mask))})
    assert response.status_code == 200, response.text
    result = response.json()
    defect = result['defects'][0]
    assert defect['source_bbox_xyxy'] == [40, 30, 85, 65]
    assert defect['pixel_measurements']['footprint_area_px2'] == 35 * 45
    assert defect['pixel_measurements']['crack'] is not None
    assert np.asarray(defect['raw_axial_depression_map']).shape == (192, 256)
    assert defect['geometry_raw']['max_raw'] > 0
    assert defect['severity']['physical_level'] is None
    assert defect['physical_measurements']['depth_d95_mm'] is None
    assert result['map_coordinates']['space'] == 'full_frame'


@pytest.mark.parametrize('kind', ['empty', 'corrupt'])
def test_invalid_masks(client, kind):
    rgb, mask = inputs()
    if kind == 'empty': mask[:] = 0
    data = b'not an image' if kind == 'corrupt' else png(mask)
    response = client.post('/api/v1/c3/analyze', files={'image': ('road.png', png(rgb)), 'mask': ('mask.png', data)})
    assert response.status_code == 422


def test_shared_automatic_handoff(client):
    rgb, mask = inputs()
    records = [{'defect_id': 'c1-c2-1', 'bbox_xyxy': [40,30,85,65], 'mask_space': 'frame', 'mask': mask.tolist(), 'score': 0.9}]
    response = client.post('/api/v1/c3/analyze', files={'image': ('road.png', png(rgb))}, data={'detections': json.dumps(records)})
    assert response.status_code == 200, response.text
    assert response.json()['defects'][0]['defect_id'] == 'c1-c2-1'


def test_unavailable_model(client, monkeypatch):
    rgb, mask = inputs()
    monkeypatch.setattr(surface_analysis, 'bridge', None)
    response = client.post('/api/v1/c3/analyze', files={'image': ('road.png', png(rgb)), 'mask': ('mask.png', png(mask))})
    assert response.status_code == 503


@pytest.mark.parametrize('format', ['JPEG', 'WEBP', 'BMP', 'PNG'])
def test_flexible_uploads(client, format):
    rgb, mask = inputs()
    small = np.repeat(mask[::2, ::2, None], 3, axis=2)
    small[0, 0] = 30
    def encode(a):
        out = io.BytesIO()
        Image.fromarray(a).save(out, format=format)
        return out.getvalue()
    response = client.post('/api/v1/c3/analyze', files={
        'image': ('road', encode(rgb)), 'mask': ('mask', encode(small)),
        'crack_mask': ('crack', encode(small)),
    })
    assert response.status_code == 200, response.text
    defect = response.json()['defects'][0]
    assert defect['pixel_measurements']['footprint_area_px2'] > 0
    assert defect['pixel_measurements']['crack'] is not None
