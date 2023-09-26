from typing import Iterable, List, Optional, Iterator, Callable
import time

import numpy as np
import torch

from data_gradients.dataset_adapters.config.typing import SupportedDataType
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.sample_preprocessor.base_sample_preprocessor import BaseSamplePreprocessor
from data_gradients.sample_preprocessor.utils.contours import get_contours
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat
from data_gradients.dataset_adapters.segmentation_adapter import SegmentationDatasetAdapter


class SegmentationSampleProcessor(BaseSamplePreprocessor):
    def __init__(
        self,
        cache_path: str,
        class_names: Optional[List[str]],
        n_classes: Optional[int],
        images_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]],
        labels_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]],
        is_batch: Optional[bool],
        class_names_to_use: List[str],
        num_image_channels: int,
        threshold_soft_labels: float,
        image_format: Optional[ImageChannelFormat],
    ):
        self.adapter = SegmentationDatasetAdapter(
            class_names=class_names,
            n_classes=n_classes,
            cache_path=cache_path,
            class_names_to_use=class_names_to_use,
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
            is_batch=is_batch,
            n_image_channels=num_image_channels,
            threshold_soft_labels=threshold_soft_labels,
        )

        self.class_names = class_names
        self.image_format = image_format
        super().__init__(data_config=self.adapter.data_config)

    def preprocess_samples(self, dataset: Iterable[SupportedDataType], split: str) -> Iterator[SegmentationSample]:
        for data in dataset:
            images, labels = self.adapter.adapt(data)
            images = np.uint8(np.transpose(images.cpu().numpy(), (0, 2, 3, 1)))
            labels = np.uint8(labels.cpu().numpy())

            for image, mask in zip(images, labels):
                contours = get_contours(mask)

                yield SegmentationSample(
                    image=image,
                    mask=mask,
                    contours=contours,
                    class_names=self.class_names,
                    split=split,
                    image_format=self.image_format,
                    sample_id=str(time.time()),
                )
