from typing import Iterable, List, Optional, Iterator, Callable
import time

import numpy as np
import torch

from data_gradients.config.data.typing import SupportedDataType
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.sample_preprocessor.base import BaseSamplePreprocessor
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter


class DetectionSamplePreprocessor(BaseSamplePreprocessor):
    def __init__(
        self,
        cache_filename: str,
        class_names: Optional[List[str]],
        n_classes: Optional[int],
        images_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]],
        labels_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]],
        is_label_first: Optional[bool],
        bbox_format: Optional[Callable[[torch.Tensor], torch.Tensor]],
        class_names_to_use: List[str],
        n_image_channels: int,
        image_format: Optional[ImageChannelFormat],
    ):
        self.adapter = DetectionDatasetAdapter(
            class_names=class_names,
            n_classes=n_classes,
            cache_filename=cache_filename,
            class_names_to_use=class_names_to_use,
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
            is_label_first=is_label_first,
            bbox_format=bbox_format,
            n_image_channels=n_image_channels,
        )
        self.class_names = class_names
        self.image_format = image_format
        super().__init__(config=self.adapter.data_config)

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
                    class_names=self.class_names,
                    split=split,
                    image_format=self.image_format,
                    sample_id=str(time.time()),
                )
