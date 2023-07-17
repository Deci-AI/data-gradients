from abc import ABC, abstractmethod
from typing import Tuple

import torch

from data_gradients.config.data.data_config import DataConfig
from data_gradients.config.data.questions import Question, text_to_yellow


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

    @staticmethod
    def ask_n_image_channels(data_config: DataConfig, images: torch.Tensor) -> int:
        """
        :param data_config: Configuration of the Dataset
        :param images:      Batch of image, in  shape (BS, C, H, W) or (BS, H, W, C)
        """
        options = {images.shape[1]: images.shape[1], images.shape[3]: images.shape[3]}
        question = Question(question=f"How many {text_to_yellow('channel(s)')} is your image made of ?", options=options)
        hint = f"This is your image shape: {images.shape}"
        # TODO: Check if this is ok or investigate better way
        return data_config.get_n_image_channels(question=question, hint=hint)
