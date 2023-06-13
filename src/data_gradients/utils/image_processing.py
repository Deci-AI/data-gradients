from typing import Tuple
import cv2
import numpy as np


def resize_in_chunks(img: np.ndarray, size: Tuple[int, int], interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize an image by chunks. This function supports any number of channels, while `cv2.resize` only supports up to 512 channels.

    :param img:             The image to resize. (H, W, C) or (H, W) expected.
    :param size:            The size to resize to, (H, W).
    :param interpolation:   The interpolation method to use.
    :return: The resized image, in (H, W, C) or (H, W).
    """
    num_channels = img.shape[2] if len(img.shape) == 3 else 1

    max_n_channels = 512
    if num_channels <= max_n_channels:
        return cv2.resize(src=img, dsize=size, interpolation=interpolation)

    chunks = []
    for index in range(0, num_channels, max_n_channels):
        chunk = img[:, :, index : index + max_n_channels]
        resized_chunk = cv2.resize(src=chunk, dsize=size, interpolation=interpolation)
        chunks.append(resized_chunk)

    return np.dstack(chunks)
