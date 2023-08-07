from typing import Iterable, List
from torch import Tensor
import numpy as np
import time

from data_gradients.utils.data_classes import DetectionSample
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat, ClassificationSample


class ClassificationBatchPreprocessor(BatchPreprocessor):
    def __init__(self, class_names: List[str], n_image_channels:int):
        """
        :param class_names: List of all class names in the dataset. The index should represent the class_id.
        """
        if n_image_channels not in [1, 3]:
            raise ValueError(f"n_image_channels should be either 1 or 3, but got {n_image_channels}")
        self.class_names = class_names
        self.n_image_channels = n_image_channels

    def preprocess(self, images: Tensor, labels: Tensor, split: str) -> Iterable[DetectionSample]:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.

        :param images:      Batch of images already formatted into (BS, C, H, W)
        :param labels:      Batch of targets (BS)
        :param split:       Name of the split (train, val, test)
        :return:            Iterable of ready to analyse detection samples.
        """
        images = np.uint8(np.transpose(images.cpu().numpy(), (0, 2, 3, 1)))

        # TODO: image_format is hard-coded here, but it should be refactored afterwards
        image_format = {1: ImageChannelFormat.GRAYSCALE, 3: ImageChannelFormat.RGB}[self.n_image_channels]

        for image, target in zip(images, labels):
            class_id = int(target)

            sample = ClassificationSample(
                image=image,
                class_id=class_id,
                class_names=self.class_names,
                split=split,
                image_format=image_format,
                sample_id=None,
            )
            sample.sample_id = str(id(sample))
            yield sample
