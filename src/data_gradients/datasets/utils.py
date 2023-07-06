import cv2
import numpy as np
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat


def load_image(path: str, channel_format: ImageChannelFormat = ImageChannelFormat.BGR) -> np.ndarray:
    """Load an image from a path in a specified format."""
    bgr_image = cv2.imread(path)
    if channel_format == ImageChannelFormat.RGB:
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    elif channel_format == ImageChannelFormat.BGR:
        return bgr_image
    elif channel_format == ImageChannelFormat.GRAYSCALE:
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError(f"Channel format {channel_format} is not supported for loading image")
