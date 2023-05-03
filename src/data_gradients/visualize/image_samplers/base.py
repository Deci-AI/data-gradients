from abc import ABC

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
        pass
