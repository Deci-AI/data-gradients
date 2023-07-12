from typing import Tuple, Optional, Callable, List

import torch
from torch import Tensor

from data_gradients.batch_processors.utils import check_all_integers
from data_gradients.batch_processors.formatters.base import BatchFormatter
from data_gradients.batch_processors.formatters.utils import ensure_images_shape, ensure_channel_first, drop_nan
from data_gradients.config.data.data_config import DetectionDataConfig, ClassificationDataConfig
from data_gradients.batch_processors.formatters.utils import DatasetFormatError


class UnsupportedClassificationBatchFormatError(DatasetFormatError):
    def __init__(self, str):
        super().__init__(str)


class ClassificationBatchFormatter(BatchFormatter):
    """Classification formatter class"""

    def __init__(
        self,
        data_config: ClassificationDataConfig,
        class_names: List[str],
        class_names_to_use: List[str],
        n_image_channels: int,
    ):
        """
        :param class_names:         List of all class names in the dataset. The index should represent the class_id.
        :param class_names_to_use:  List of class names that we should use for analysis.
        :param n_image_channels:    Number of image channels (3 for RGB, 1 for Gray Scale, ...)
        """
        self.data_config = data_config

        class_names_to_use = set(class_names_to_use)
        self.class_ids_to_use = [class_id for class_id, class_name in enumerate(class_names) if class_name in class_names_to_use]

        self.n_image_channels = n_image_channels

    def format(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for detection.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in (BS) format
        :return:
            - images: Batch of images already formatted into (BS, C, H, W)
            - labels: Batch of targets (BS)
        """

        images = ensure_channel_first(images, n_image_channels=self.n_image_channels)
        images = ensure_images_shape(images, n_image_channels=self.n_image_channels)
        labels = self.ensure_labels_shape(images=images, labels=labels)

        if 0 <= images.min() and images.max() <= 1:
            images *= 255
            images = images.to(torch.uint8)
        elif images.min() < 0:  # images were normalized with some unknown mean and std
            images -= images.min()
            images /= images.max()
            images *= 255
            images = images.to(torch.uint8)

        return images, labels

    @staticmethod
    def ensure_labels_shape(labels: Tensor, images: Tensor) -> Tensor:
        """Make sure that the labels have the correct shape, i.e. (BS)."""
        if torch.is_floating_point(labels):
            raise UnsupportedClassificationBatchFormatError("Labels should be integers")

        if labels.ndim != 1:
            raise UnsupportedClassificationBatchFormatError("Labels should be 1D tensor")

        if len(labels) != len(images):
            raise UnsupportedClassificationBatchFormatError("Labels and images should have the same length")

        return labels
