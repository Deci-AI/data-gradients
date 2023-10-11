from abc import ABC, abstractmethod
from typing import Tuple

import torch
from data_gradients.dataset_adapters.config.questions import OpenEndedQuestion


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
            question = OpenEndedQuestion(
                question="Please describe your image channels?",
            )
            hint = (
                f"Image shape: {images.shape}\n\n"
                "Options:\n"
                "  - `RGB`\n"
                "  - `BGR`\n"
                "  - `L` (Grayscale)\n"
                "  - `LAB`\n"
                "Note: If your image has extra channels, you please add a `O` for each of them.\n\n"
                "E.g. If you have image has 4 (Red, Green, Blue, Depth), you should enter `RGBO` (in the right order!)"
            )

            image_channels = self.data_config.get_image_channels(question=question, hint=hint)
            self._n_image_channels = len(image_channels)
        return self._n_image_channels
