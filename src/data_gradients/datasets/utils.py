import cv2
import numpy as np
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat


def load_image(path: str, channel_format: ImageChannelFormat = ImageChannelFormat.BGR) -> np.ndarray:
    """Load an image from a path in a specified format."""
    if channel_format == ImageChannelFormat.BGR:
        return cv2.imread(path, cv2.IMREAD_COLOR)
    elif channel_format == ImageChannelFormat.RGB:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif channel_format == ImageChannelFormat.GRAYSCALE:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif channel_format == ImageChannelFormat.UNCHANGED:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        raise NotImplementedError(f"Channel format {channel_format} is not supported for loading image")
