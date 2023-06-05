import dataclasses
from enum import Enum
from typing import List

import numpy as np

from data_gradients.utils.data_classes.contour import Contour


class ImageChannelFormat(Enum):
    RGB = "RGB"
    BGR = "BGR"
    GRAYSCALE = "GRAYSCALE"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass
class ImageSample:
    """
    This is a dataclass that represents a single sample of the dataset where input to the model is a single image.

    :attr sample_id:The unique identifier of the sample. Could be the image path or the image name.
    :attr split:    The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:    np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    """

    sample_id: str
    split: str
    image: np.ndarray
    image_format: ImageChannelFormat

    def __repr__(self):
        return f"ImageSample(sample_id={self.sample_id}, image={self.image.shape}, format={self.image_format})"


@dataclasses.dataclass
class SegmentationSample(ImageSample):
    """
    This is a dataclass that represents a single sample of the dataset where input to the model is a single image and
    the target is a semantic segmentation mask.

    :attr sample_id:        The unique identifier of the sample. Could be the image path or the image name.
    :attr split:            The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:            np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    :attr mask:             np.ndarray of shape [N, H, W] representing one-hot encoded mask for each class.
    :attr contours:         A list of contours for each class in the mask.
    :attr class_names:      List of all class names in the dataset. The index should represent the class_id.
    """

    mask: np.ndarray

    contours: List[List[Contour]]
    class_names: List[str]

    def __repr__(self):
        return f"SegmentationSample(sample_id={self.sample_id}, image={self.image.shape}, mask={self.mask.shape})"


@dataclasses.dataclass
class DetectionSample(ImageSample):
    """
    This is a dataclass that represents a single sample of the dataset where input to the model is a single image and
    the target is a semantic segmentation mask.

    :attr sample_id:    The unique identifier of the sample. Could be the image path or the image name.
    :attr split:        The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:        np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    :attr bboxes_xyxy:  np.ndarray of shape [N, 4] (X, Y, X, Y)
    :attr class_ids:    np.ndarray of shape [N, ]
    :attr class_names:  List of all class names in the dataset. The index should represent the class_id.
    """

    bboxes_xyxy: np.ndarray
    class_ids: np.ndarray
    class_names: List[str]

    def __repr__(self):
        return f"DetectionSample(sample_id={self.sample_id}, image={self.image.shape}, bboxes_xyxy={self.bboxes_xyxy.shape}, class_ids={self.class_ids.shape})"
