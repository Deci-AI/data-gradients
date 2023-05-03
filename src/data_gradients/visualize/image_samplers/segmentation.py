import torch
import torchvision

from data_gradients.utils import SegmentationBatchData
from data_gradients.visualize.image_samplers.base import ImageSampleManager


class SegmentationImageSampleManager(ImageSampleManager):
    def update(self, data: SegmentationBatchData) -> None:
        for image, label in zip(data.images, data.labels):
            if len(self.samples) < self.n_samples:
                self.samples.append(prepare_segmentation_image(image=image, label=label))


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
