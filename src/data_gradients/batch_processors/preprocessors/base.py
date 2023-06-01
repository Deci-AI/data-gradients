from abc import ABC, abstractmethod
from typing import Iterable

import torch

from data_gradients.utils.data_classes import ImageSample


class BatchPreprocessor(ABC):
    """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing."""

    @abstractmethod
    def preprocess(self, images: torch.Tensor, labels: torch.Tensor, split: str) -> Iterable[ImageSample]:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.

        :param images:      Batch of images already formatted into (BS, C, H, W)
        :param labels:      Batch of labels already formatted into format relevant for current task (detection, segmentation, classification).
        :param split:       Name of the split (train, val, test)
        :return:            Ready to analyse batch object, that depends on the current task (detection, segmentation, classification).
        """
        pass
