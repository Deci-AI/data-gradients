from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from data_gradients.utils.data_classes import ImageSample


class ImageSampleManager(ABC):
    """Manage a collection of preprocessed image samples."""

    def __init__(self, n_samples: int):
        """
        :param n_samples: The maximum number of samples to be collected.
        """
        self.n_samples = n_samples
        self.samples: List[np.ndarray] = []

    def update(self, sample: ImageSample) -> None:
        """Update the internal collection of samples with new samples from the given batch data.

        :param data: The batch data containing images and labels.
        """
        if len(self.samples) < self.n_samples:
            self.samples.append(self.prepare_image(sample))

    @abstractmethod
    def prepare_image(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Prepare an individual image for visualization.

        :param image:   Input image tensor.
        :param label:   Input Label
        :return:        The preprocessed image tensor.
        """
        pass
