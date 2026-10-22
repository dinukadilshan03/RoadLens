"""Verify real-mask preparation rather than visually plausible invented geometry."""

import numpy as np
import pytest
from backend.app.services.c3_surface import prepare_relative_surface


def test_surface_uses_only_paired_mask_and_negative_depression():
    raw = np.full((192, 256), 999.0)
    mask = np.zeros((384, 512), dtype=bool)
    mask[140:180, 200:240] = True
    raw[70:90, 100:120] = 4
    surface = prepare_relative_surface(raw, mask)
    z = np.asarray(surface["z"])
    assert z.min() == -1
    assert z.max() == 0  # Surrounding road stays at the reference plane.
    assert np.array_equal(z, -np.asarray(surface["depression"]))
    assert surface["native_statistics"] == {"max_raw": 4, "mean_raw": 4, "d95_raw": 4}
    assert surface["physical_scale_verified"] is False
    assert surface["sampling_step"] == 2
    assert surface["native_shape"] == [192, 256]
    assert len(surface["x"]) == z.shape[1]
    assert len(surface["y"]) == z.shape[0]
    assert 0 <= min(surface["x"]) <= max(surface["x"]) <= 100


def test_flat_surface_does_not_invent_a_pit():
    mask = np.zeros((192, 256), dtype=bool)
    mask[30:80, 40:90] = True
    surface = prepare_relative_surface(np.zeros(mask.shape), mask)
    assert not surface["has_depression"]
    assert np.asarray(surface["z"]).sum() == 0


def test_surface_uses_original_model_grid_display_proportions():
    raw = np.ones((192, 256))
    portrait = prepare_relative_surface(raw, np.ones((800, 400)))
    landscape = prepare_relative_surface(raw, np.ones((400, 800)))
    assert portrait["aspect_ratio"]["x"] / portrait["aspect_ratio"]["y"] == pytest.approx(256 / 192)
    assert landscape["aspect_ratio"] == portrait["aspect_ratio"]
    assert portrait["aspect_ratio"]["z"] == 0.5


def test_decimation_does_not_erase_a_narrow_depression():
    mask = np.zeros((192, 256), dtype=bool)
    mask[40:48, 50:58] = True
    raw = np.zeros(mask.shape)
    raw[41, 51] = 5  # The script's every-second-pixel sample misses this point.
    surface = prepare_relative_surface(raw, mask)
    assert surface["sampling_step"] == 1
    assert np.asarray(surface["z"]).min() == -1


def test_missing_model_grid_region_is_unavailable():
    assert prepare_relative_surface(np.zeros((192, 256)), np.zeros((400, 600))) is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -2])
def test_reject_invalid_model_predictions(value):
    raw = np.zeros((192, 256))
    raw[0, 0] = value
    with pytest.raises(ValueError):
        prepare_relative_surface(raw, np.ones_like(raw))
