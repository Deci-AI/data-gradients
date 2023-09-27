from typing import Iterable, Optional, Iterator
import time

import numpy as np

from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.sample_preprocessor.base_sample_preprocessor import AbstractSamplePreprocessor
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter
from data_gradients.dataset_adapters.config import DetectionDataConfig


class DetectionSamplePreprocessor(AbstractSamplePreprocessor):
    def __init__(
        self,
        data_config: DetectionDataConfig,
        n_image_channels: int,
        image_format: Optional[ImageChannelFormat],
    ):
        self.data_config = data_config
        self.image_format = image_format

        self.adapter = DetectionDatasetAdapter(data_config=data_config, n_image_channels=n_image_channels)
        super().__init__(data_config=data_config)

    def preprocess_samples(self, dataset: Iterable[SupportedDataType], split: str) -> Iterator[DetectionSample]:
        for data in dataset:
            images, labels = self.adapter.adapt(data)
            images = np.uint8(np.transpose(images.cpu().numpy(), (0, 2, 3, 1)))

            for image, target in zip(images, labels):
                target = target.cpu().numpy().astype(int)
                class_ids, bboxes_xyxy = target[:, 0], target[:, 1:]

                yield DetectionSample(
                    image=image,
                    class_ids=class_ids,
                    bboxes_xyxy=bboxes_xyxy,
                    class_names=self.data_config.get_class_names(),
                    split=split,
                    image_format=self.image_format,
                    sample_id=str(time.time()),
                )
