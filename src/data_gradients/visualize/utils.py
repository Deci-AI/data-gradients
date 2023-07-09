import cv2
import numpy as np


def resize_keep_aspect_ratio(image: np.ndarray, target_shape: tuple):
    """
    Resizes a single image to fit within a specified width while maintaining the aspect ratio.
    If the resulting height is below 400 pixels, the image won't be padded. The width is padded
    with center alignment, and the top is padded with white color to maintain the aspect ratio.

    Args:
        image (np.ndarray): The input image.
        target_shape (tuple): A tuple (width, height) specifying the desired dimensions.

    Returns:
        np.ndarray: The resized and padded image.

    """
    image_height, image_width = image.shape[:2]
    target_width, target_height = target_shape

    scale_factor = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    x = int((target_width - new_width) / 2)
    y = int(target_height - new_height)
    canvas[y : y + new_height, x : x + new_width] = resized_image

    return canvas
