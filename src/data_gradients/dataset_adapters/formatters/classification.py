from typing import Tuple, List

import torch
from torch import Tensor

from data_gradients.dataset_adapters.formatters.base import BatchFormatter
from data_gradients.dataset_adapters.formatters.utils import DatasetFormatError, check_images_shape
from data_gradients.dataset_adapters.formatters.utils import ensure_channel_first
from data_gradients.dataset_adapters.config.data_config import ClassificationDataConfig
from data_gradients.utils.data_classes.data_samples import Image
from logging import getLogger

logger = getLogger(__name__)


class UnsupportedClassificationBatchFormatError(DatasetFormatError):
    def __init__(self, str):
        super().__init__(str)


class ClassificationBatchFormatter(BatchFormatter):
    """Classification formatter class"""

    def __init__(self, data_config: ClassificationDataConfig):
        self.data_config = data_config

        if data_config.get_class_names_to_use() != data_config.get_class_names():
            logger.warning("Classification task does NOT support class filtering, yet `class_names_to_use` was set. This will parameter will be ignored.")

        super().__init__(data_config=data_config)

    def format(self, images: Tensor, labels: Tensor) -> Tuple[List[Image], Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for detection.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in (BS) format
        :return:
            - images: Batch of images already formatted into (BS, C, H, W)
            - labels: Batch of targets (BS)
        """

        if not self.check_is_batch(images=images, labels=labels):
            images = images.unsqueeze(0)
            labels = labels.unsqueeze(0)

        images = ensure_channel_first(images, n_image_channels=self.get_n_image_channels(images=images))
        images = check_images_shape(images, n_image_channels=self.get_n_image_channels(images=images))

        labels = self.ensure_labels_shape(images=images, labels=labels)
        image_formatted = self._format_images(images)
        return image_formatted, labels

    def check_is_batch(self, images: Tensor, labels: Tensor) -> bool:
        if images.ndim == 4:
            self.data_config.is_batch = True
            return self.data_config.is_batch
        elif images.ndim == 2 or labels.ndim == 1:
            self.data_config.is_batch = False
            return self.data_config.is_batch
        else:
            hint = f"    - Image shape: {images.shape}\n    - Label shape:  {labels.shape}"
            return self.data_config.get_is_batch(hint=hint)

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
