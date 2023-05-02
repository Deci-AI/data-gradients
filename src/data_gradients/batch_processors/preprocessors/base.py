from abc import ABC, abstractmethod

import torch

from data_gradients.utils import BatchData


class BatchPreprocessor(ABC):
    """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing."""

    @abstractmethod
    def preprocess(self, images: torch.Tensor, labels: torch.Tensor) -> BatchData:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.

        :param images:  Batch of images already formatted into (BS, C, H, W)
        :param labels:  Batch of labels already formatted into format relevant for current task (detection, segmentation, classification).
        :return:        Ready to analyse batch object, that depends on the current task (detection, segmentation, classification).
        """
        pass
