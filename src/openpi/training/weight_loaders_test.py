import numpy as np

from openpi.training import weight_loaders


def test_adapt_rgbd_stem():
    rgb_stem = np.arange(
        2 * 2 * 3 * 4,
        dtype=np.float32,
    ).reshape(2, 2, 3, 4)

    loaded = {
        "PaliGemma": {
            "img": {
                "embedding": {
                    "kernel": rgb_stem,
                }
            }
        },
        "other": {"kernel": np.array([1.0], dtype=np.float32)},
    }
    reference = {
        "PaliGemma": {
            "img": {
                "embedding": {
                    "kernel": np.empty(
                        (2, 2, 4, 4),
                        dtype=np.float32,
                    ),
                }
            }
        }
    }

    adapted = weight_loaders._adapt_rgbd_stem(
        loaded,
        reference,
    )
    stem = adapted["PaliGemma"]["img"]["embedding"]["kernel"]

    np.testing.assert_array_equal(stem[..., :3, :], rgb_stem)
    np.testing.assert_array_equal(stem[..., 3, :], 0)
    np.testing.assert_array_equal(adapted["other"]["kernel"], loaded["other"]["kernel"])
    assert stem.shape == (2, 2, 4, 4)


def test_adapt_rgbd_stem_keeps_matching_checkpoint():
    stem = np.ones((2, 2, 4, 4), dtype=np.float32)
    params = {"PaliGemma": {"img": {"embedding": {"kernel": stem}}}}

    assert weight_loaders._adapt_rgbd_stem(params, params) is params
