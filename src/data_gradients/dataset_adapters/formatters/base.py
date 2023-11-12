from abc import ABC, abstractmethod
from typing import Tuple, List

import torch
from data_gradients.dataset_adapters.formatters.utils import check_images_shape, ensure_channel_first
from data_gradients.utils.data_classes.data_samples import Image


class BatchFormatter(ABC):
    def __init__(self, data_config):
        self.data_config = data_config
        self._n_image_channels = None

    @abstractmethod
    def format(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[List[Image], torch.Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for a given task.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in task-dependant format
        :return:
            - images: List of images
            - labels: Batch of labels already formatted into format relevant for current task (detection, segmentation, classification).
        """
        pass

    def get_n_image_channels(self, images: torch.Tensor) -> int:
        """Get the number of image channels in the batch. If not set yet, it will be asked to the user."""
        if self._n_image_channels is None:
            image_channels = self.data_config.get_image_channels(image=images[0])
            self._n_image_channels = len(image_channels)
        return self._n_image_channels

    def _format_images(self, images: torch.Tensor) -> List[Image]:
        """Format images into a list of Image in a standard format."""
        images = ensure_channel_first(images, n_image_channels=self.get_n_image_channels(images=images))
        images = check_images_shape(images, n_image_channels=self.get_n_image_channels(images=images))

        image_format = self.data_config.get_image_normalizer(images=images)
        return [Image(data=image, format=image_format).to_uint8() for image in images]
