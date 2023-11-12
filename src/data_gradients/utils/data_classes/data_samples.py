import dataclasses
from typing import List, Dict, Union

import numpy as np
import torch

from data_gradients.utils.data_classes.contour import Contour
from data_gradients.utils.data_classes.image_channels import ImageChannels
from data_gradients.dataset_adapters.formatters.utils import ImageFormat, Uint8ImageFormat, FloatImageFormat, ScaledFloatImageFormat
from dataclasses import dataclass


@dataclass
class Image:
    data: Union[torch.Tensor, np.ndarray]
    format: ImageFormat

    def __post_init__(self):
        self.data = self.format.convert_image_to_float(self.data)

    def to_uint8(self) -> "Image":
        return self._to_format(target_format=Uint8ImageFormat())

    def to_float(self) -> "Image":
        return self._to_format(target_format=FloatImageFormat())

    def to_scaled_float(self, mean: List[float], std: List[float]) -> "Image":
        return self._to_format(target_format=ScaledFloatImageFormat(mean=mean, std=std))

    def _to_format(self, target_format: ImageFormat) -> "Image":
        if isinstance(target_format, type(self.format)):
            return self
        else:
            float_image = self.format.convert_image_to_float(images=self.data)
            return Image(data=target_format.convert_image_from_float(images=float_image), format=target_format)

    @property
    def as_numpy(self) -> np.ndarray:
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return self.data

    @property
    def as_torch(self) -> torch.Tensor:
        if isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data)
        return self.data


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
    image_channels: ImageChannels  # TODO: rename

    def __repr__(self):
        return f"ImageSample(sample_id={self.sample_id}, image={self.image.shape}, format={self.image_channels})"

    @property
    def image_as_rgb(self) -> np.ndarray:
        return self.image_channels.convert_image_to_rgb(image=self.image)

    @property
    def image_channels_to_visualize(self) -> np.ndarray:
        return self.image_channels.get_channels_to_visualize(image=self.image)

    @property
    def image_mean_intensity(self) -> float:
        return self.image_channels.compute_mean_image_intensity(image=self.image)


@dataclasses.dataclass
class SegmentationSample(ImageSample):
    """
    This is a dataclass that represents a single sample of the dataset where input to the model is a single image and
    the target is a semantic segmentation mask.

    :attr sample_id:        The unique identifier of the sample. Could be the image path or the image name.
    :attr split:            The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:            np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    :attr mask:             np.ndarray of shape [H, W], categorical representation of the mask.
    :attr contours:         A list of contours for each class in the mask.
    :attr class_names:      List of all class names in the dataset. The index should represent the class_id.
    """

    mask: np.ndarray

    contours: List[List[Contour]]
    class_names: Dict[int, str]

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
    class_names: Dict[int, str]

    def __repr__(self):
        return f"DetectionSample(sample_id={self.sample_id}, image={self.image.shape}, bboxes_xyxy={self.bboxes_xyxy.shape}, class_ids={self.class_ids.shape})"


@dataclasses.dataclass
class ClassificationSample(ImageSample):
    """
    This is a dataclass that represents a single classification sample of the dataset where input to the model is
    a single image and the target is an image label.

    :attr sample_id:    The unique identifier of the sample. Could be the image path or the image name.
    :attr split:        The name of the dataset split. Could be "train", "val", "test", etc.
    :attr image:        np.ndarray of shape [H,W,C] - The image as a numpy array with channels last.
    :attr class_label:  Class label (int)
    :attr class_names:  List of all class names in the dataset. The index should represent the class_id.
    """

    class_id: int
    class_names: Dict[int, str]

    def __repr__(self):
        return f"DetectionSample(sample_id={self.sample_id}, image={self.image.shape}, label={self.class_id})"
