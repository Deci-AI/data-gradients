from abc import ABC, abstractmethod
from typing import Tuple

import torch
from data_gradients.dataset_adapters.config.questions import FixedOptionsQuestion


class BatchFormatter(ABC):
    def __init__(self, data_config):
        self.data_config = data_config
        self._n_image_channels = None

    @abstractmethod
    def format(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate batch images and labels format, and ensure that they are in the relevant format for a given task.

        :param images: Batch of images, in (BS, ...) format
        :param labels: Batch of labels, in task-dependant format
        :return:
            - images: Batch of images already formatted into (BS, C, H, W)
            - labels: Batch of labels already formatted into format relevant for current task (detection, segmentation, classification).
        """
        pass

    def get_n_image_channels(self, images: torch.Tensor) -> int:
        """Get the number of image channels in the batch. If not set yet, it will be asked to the user."""
        if self._n_image_channels is None:
            question = FixedOptionsQuestion(
                question="Which dimension corresponds the image channel? ",
                options={i: images.shape[i] for i in range(len(images.shape))},
            )
            hint = f"Image shape: {images.shape}"
            self._n_image_channels = self.data_config.get_n_image_channels(question=question, hint=hint)
        return self._n_image_channels
