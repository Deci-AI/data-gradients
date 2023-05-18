import numpy as np
import torch

from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.image_samplers.base import ImageSampleManager


class SegmentationImageSampleManager(ImageSampleManager):
    def prepare_image(self, sample: SegmentationSample) -> np.ndarray:
        return prepare_segmentation_image(image=sample.image, label=sample.mask)


def prepare_segmentation_image(image: np.ndarray, label: np.ndarray) -> torch.Tensor:
    """Combine image and label to a single image with the original image to the left and the mask to the right.

    :param image:   Input image tensor.
    :param label:   Input Label.
    :return:        The preprocessed image tensor.
    """
    # TODO: Add overlay of contours
    return image
