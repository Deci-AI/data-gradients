from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BatchFormatter(ABC):
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
