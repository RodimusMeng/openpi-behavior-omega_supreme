import numpy as np

from openpi.policies import b1k_policy


def test_normalize_depth_meter_millimeter_equivalence():
    depth_m = np.asarray([[0.01, 0.1], [1.0, 10.0]], dtype=np.float32)

    normalized_m = b1k_policy.normalize_depth(depth_m, input_unit="m")
    normalized_mm = b1k_policy.normalize_depth(depth_m * 1000.0, input_unit="mm")

    np.testing.assert_allclose(normalized_m, normalized_mm, atol=1e-6)
    np.testing.assert_allclose(normalized_m[..., 0], [[-1.0, -1.0 / 3.0], [1.0 / 3.0, 1.0]], atol=1e-6)


def test_normalize_depth_layouts_and_invalid_values():
    depth_hwc = np.asarray([[[1000.0], [np.nan]], [[0.0], [np.inf]]], dtype=np.float32)
    depth_chw = np.moveaxis(depth_hwc, -1, 0)

    normalized_hwc = b1k_policy.normalize_depth(depth_hwc, input_unit="mm")
    normalized_chw = b1k_policy.normalize_depth(depth_chw, input_unit="mm")

    np.testing.assert_allclose(normalized_hwc, normalized_chw)
    assert normalized_hwc.shape == (2, 2, 1)
    assert np.isfinite(normalized_hwc).all()
    np.testing.assert_allclose(normalized_hwc[0, 0, 0], 1.0 / 3.0, atol=1e-6)
    np.testing.assert_allclose(normalized_hwc[[0, 1, 1], [1, 0, 1], 0], 1.0, atol=1e-6)


def test_prepare_rgbd_preserves_pairing_and_channel_contract():
    rgb = np.asarray([[[0, 127, 255], [255, 127, 0]]], dtype=np.uint8)
    depth_mm = np.asarray([[1000.0, 100.0]], dtype=np.float32)

    rgbd = b1k_policy.prepare_rgbd(rgb, depth_mm, depth_unit="mm")

    assert rgbd.shape == (1, 2, 4)
    np.testing.assert_allclose(rgbd[..., :3], rgb.astype(np.float32) / 127.5 - 1.0)
    np.testing.assert_allclose(rgbd[..., 3], [[1.0 / 3.0, -1.0 / 3.0]], atol=1e-6)
