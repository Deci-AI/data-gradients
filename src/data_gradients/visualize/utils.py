from typing import Tuple
import cv2
import numpy as np


def resize_and_align_bottom_center(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resizes an image while maintaining its aspect ratio, and aligns it at the bottom center on a canvas of the target size.

    :param image:           Input image to resize and center.
    :param target_shape:    Desired output shape as (height, width).
    :return:                Output image, which is the input image resized, centered horizontally, and aligned at the bottom on a canvas of the target size.
    """
    image_height, image_width = image.shape[:2]
    target_height, target_width = target_shape

    scale_factor = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    x = int((target_width - new_width) / 2)
    y = int(target_height - new_height)
    canvas[y : y + new_height, x : x + new_width] = resized_image

    return canvas
