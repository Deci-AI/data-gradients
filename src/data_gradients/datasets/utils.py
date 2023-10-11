import cv2
import numpy as np


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image from a path in a RGB.

    :return: The image as a numpy array. (H, W, 3)
    """
    return cv2.imread(path, cv2.IMREAD_COLOR)
