from typing import Iterable, Optional, Iterator
import time

import numpy as np

from data_gradients.dataset_adapters.config.typing import SupportedDataType
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.sample_preprocessor.base_sample_preprocessor import AbstractSamplePreprocessor
from data_gradients.sample_preprocessor.utils.contours import get_contours
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat
from data_gradients.dataset_adapters.segmentation_adapter import SegmentationDatasetAdapter
from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig


class SegmentationSampleProcessor(AbstractSamplePreprocessor):
    def __init__(self, data_config: SegmentationDataConfig, threshold_soft_labels: float, image_format: Optional[ImageChannelFormat]):
        self.data_config = data_config
        self.image_format = image_format

        self.adapter = SegmentationDatasetAdapter(data_config=data_config, threshold_soft_labels=threshold_soft_labels)
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
                    class_names=self.data_config.class_names,
                    split=split,
                    image_format=self.image_format,
                    sample_id=str(time.time()),
                )
