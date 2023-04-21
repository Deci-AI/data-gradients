from abc import ABC, abstractmethod
import torch
import torchvision

from data_gradients.utils import BatchData


class ImageVisualizer(ABC):
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.samples = []

    def update(self, data: BatchData) -> None:
        for image, label in zip(data.images, data.labels):
            if len(self.samples) < self.n_samples:
                self.samples.append(self.prepare_image(image=image, label=label))

    @abstractmethod
    def prepare_image(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pass


class SegmentationImageVisualizer(ImageVisualizer):
    """
    Semantic Segmentation task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """

    def prepare_image(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return prepare_segmentation_image(image=image, label=label)


def prepare_segmentation_image(image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
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
