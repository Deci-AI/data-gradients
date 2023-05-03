from abc import ABC, abstractmethod

import torch

from data_gradients.utils import BatchData


class ImageSampleManager(ABC):
    """Manage a collection of preprocessed image samples."""

    def __init__(self, n_samples: int):
        """
        :param n_samples: The maximum number of samples to be collected.
        """
        self.n_samples = n_samples
        self.samples = []

    def update(self, data: BatchData) -> None:
        """Update the internal collection of samples with new samples from the given batch data.

        :param data: The batch data containing images and labels.
        """
        for image, label in zip(data.images, data.labels):
            if len(self.samples) < self.n_samples:
                self.samples.append(self.prepare_image(image=image, label=label))

    @abstractmethod
    def prepare_image(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Prepare an individual image for visualization.

        :param image:   Input image tensor.
        :param label:   Input Label
        :return:        The preprocessed image tensor.
        """
        pass
