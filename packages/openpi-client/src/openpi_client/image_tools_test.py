import numpy as np

import openpi_client.image_tools as image_tools


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = np.zeros((2, 10, 10, 3), dtype=np.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = np.zeros((3, 30, 30, 3), dtype=np.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = np.zeros((1, 50, 50, 3), dtype=np.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 4: Resize image with odd-numbered padding
    images = np.zeros((1, 256, 320, 3), dtype=np.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)


def test_resize_with_pad_float32_rgbd():
    pattern = np.zeros((8, 8), dtype=np.float32)
    pattern[2:6, 2:6] = 1.0
    image = np.stack([pattern * scale for scale in (1.0, 2.0, 3.0, 4.0)], axis=-1)

    resized = image_tools.resize_with_pad(image[None], 4, 4)

    assert resized.shape == (1, 4, 4, 4)
    assert resized.dtype == np.float32
    for channel, scale in enumerate((1.0, 2.0, 3.0, 4.0)):
        np.testing.assert_allclose(resized[0, ..., channel], resized[0, ..., 0] * scale)
