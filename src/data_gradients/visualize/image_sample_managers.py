from abc import ABC, abstractmethod
import torch
import torchvision

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


class SegmentationImageSampleManager(ImageSampleManager):
    def prepare_image(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return prepare_segmentation_image(image=image, label=label)


def prepare_segmentation_image(image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Combine image and label to a single image with the original image to the left and the mask to the right.

    :param image:   Input image tensor.
    :param label:   Input Label.
    :return:        The preprocessed image tensor.
    """
    # Just in case label is different from image
    if label.shape[2:] != image.shape[2:]:
        resize = torchvision.transforms.Resize(label.shape[-2:])
        image = resize(image)

    idxs = torch.arange(len(label)).view(len(label), 1, 1) + 1  # +1 to avoid 'killing' channel zero
    label = idxs * label
    label = torch.sum(label, dim=0)
    label = torch.ceil((label * 255) / torch.max(label) + 1)  # normalize such that max(label) goes to 255
    label = label.repeat(3, 1, 1)  # to rgb (with same values -> grayscale)
    return torch.cat((image.float(), label), dim=-1)
