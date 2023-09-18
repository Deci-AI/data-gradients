from typing import Iterable, List, Optional, Iterator, Callable
import time

import numpy as np
import torch

from data_gradients.config.data.typing import SupportedDataType
from data_gradients.sample_preprocessor.base_sample_preprocessor import BaseSamplePreprocessor
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat, ClassificationSample
from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter


class ClassificationSamplePreprocessor(BaseSamplePreprocessor):
    def __init__(
        self,
        cache_path: str,
        class_names: Optional[List[str]],
        n_classes: Optional[int],
        images_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]],
        labels_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]],
        class_names_to_use: List[str],
        n_image_channels: int,
        image_format: Optional[ImageChannelFormat],
    ):
        if n_image_channels not in [1, 3]:
            raise ValueError(f"n_image_channels should be either 1 or 3, but got {n_image_channels}")

        self.adapter = ClassificationDatasetAdapter(
            class_names=class_names,
            n_classes=n_classes,
            cache_path=cache_path,
            class_names_to_use=class_names_to_use,
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
            n_image_channels=n_image_channels,
        )
        self.class_names = class_names
        self.n_image_channels = n_image_channels
        self.image_format = image_format
        super().__init__(config=self.adapter.data_config)

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
                    class_names=self.class_names,
                    split=split,
                    image_format=self.image_format,
                    sample_id=str(time.time()),
                )
                yield sample
