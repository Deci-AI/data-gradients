from typing import Any, Tuple, List

import torch

from data_gradients.batch_processors.adapters.common_extractors.base import BaseDatasetExtractor
from data_gradients.batch_processors.adapters.common_extractors.coco import (
    check_coco_dataset_format,
    coco_image_type,
    coco_annotation_type,
    coco_detection_labels_extractor,
)
from data_gradients.batch_processors.adapters.utils import to_torch


class CocoDetectionExtractor(BaseDatasetExtractor):
    def is_valid(self, data: Any) -> bool:
        return check_coco_dataset_format(data)

    def images_extractor(self, data: Tuple[coco_image_type, List[coco_annotation_type]]) -> torch.Tensor:
        return to_torch(data[0])

    def labels_extractor(self, data: Tuple[coco_image_type, List[coco_annotation_type]]) -> torch.Tensor:
        return coco_detection_labels_extractor(coco_sample=data)
