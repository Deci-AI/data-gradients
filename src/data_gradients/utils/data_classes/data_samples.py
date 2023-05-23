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

    Properties:
        sample_id: str - The unique identifier of the sample. Could be the image path or the image name.
        split: str - The name of the dataset split. Could be "train", "val", "test", etc.
        image: np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
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

    Properties:
        sample_id: str - The unique identifier of the sample. Could be the image path or the image name.
        split: str - The name of the dataset split. Could be "train", "val", "test", etc.
        image: np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
        mask: np.ndarray of shape [N, H, W] representing one-hot encoded mask for each class.
        contours: List[List[Contour]] - A list of contours for each class in the mask.
    """

    mask: np.ndarray

    contours: List[List[Contour]]

    def __repr__(self):
        return f"SegmentationSample(sample_id={self.sample_id}, image={self.image.shape}, mask={self.mask.shape})"


@dataclasses.dataclass
class DetectionSample(ImageSample):
    """
    This is a dataclass that represents a single sample of the dataset where input to the model is a single image and
    the target is a semantic segmentation mask.

    Properties:
        sample_id: str - The unique identifier of the sample. Could be the image path or the image name.
        split: str - The name of the dataset split. Could be "train", "val", "test", etc.
        image: np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
        target: np.ndarray of shape [N, 5] (Label, X, Y, X, Y)
    """

    target: np.ndarray

    def __repr__(self):
        return f"DetectionSample(sample_id={self.sample_id}, image={self.image.shape}, target={self.target.shape})"
