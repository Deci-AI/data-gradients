from typing import Iterable, Optional, Iterator
import time

import numpy as np

from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType
from data_gradients.sample_preprocessor.base_sample_preprocessor import AbstractSamplePreprocessor
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat, ClassificationSample
from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter
from data_gradients.dataset_adapters.config.data_config import ClassificationDataConfig


class ClassificationSamplePreprocessor(AbstractSamplePreprocessor):
    def __init__(self, data_config: ClassificationDataConfig, n_image_channels: int, image_format: Optional[ImageChannelFormat]):
        if n_image_channels not in [1, 3]:
            raise ValueError(f"n_image_channels should be either 1 or 3, but got {n_image_channels}")

        self.data_config = data_config
        self.n_image_channels = n_image_channels
        self.image_format = image_format

        self.adapter = ClassificationDatasetAdapter(data_config=data_config, n_image_channels=n_image_channels)
        super().__init__(data_config=self.adapter.data_config)

    def preprocess_samples(self, dataset: Iterable[SupportedDataType], split: str) -> Iterator[ClassificationSample]:
        for data in dataset:
            images, labels = self.adapter.adapt(data)
            images = np.uint8(np.transpose(images.cpu().numpy(), (0, 2, 3, 1)))

            if self.image_format is None:
                self.image_format = {1: ImageChannelFormat.GRAYSCALE, 3: ImageChannelFormat.RGB}[self.n_image_channels]

            for image, target in zip(images, labels):
                class_id = int(target)

                sample = ClassificationSample(
                    image=image,
                    class_id=class_id,
                    class_names=self.data_config.class_names,
                    split=split,
                    image_format=self.image_format,
                    sample_id=str(time.time()),
                )
                yield sample
