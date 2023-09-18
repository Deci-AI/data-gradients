from typing import List, Optional, Callable

import torch

from data_gradients.config.data.typing import SupportedDataType

from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.dataset_adapters.formatters.segmentation import SegmentationBatchFormatter
from data_gradients.config.data.data_config import SegmentationDataConfig


class SegmentationDatasetAdapter(BaseDatasetAdapter):
    """Wrap a segmentation dataset so that it would return standardized tensors.

    :param cache_path:              The filename of the cache file.
    :param n_classes:               The number of classes.
    :param class_names:             List of class names.
    :param class_names_to_use:      List of class names to use.
    :param images_extractor:        Callable function for extracting images.
    :param labels_extractor:        Callable function for extracting labels.
    :param n_image_channels:        Number of image channels.
    :param threshold_soft_labels:   Soft labels threshold.
    :param data_config:             Instance of DetectionDataConfig class that manages dataset/dataloader configurations.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        n_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        class_names_to_use: Optional[List[str]] = None,
        images_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]] = None,
        labels_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]] = None,
        n_image_channels: int = 3,
        threshold_soft_labels: float = 0.5,
        data_config: Optional[SegmentationDataConfig] = None,
    ):
        class_names = self.resolve_class_names(class_names=class_names, n_classes=n_classes)
        class_names_to_use = self.resolve_class_names_to_use(class_names=class_names, class_names_to_use=class_names_to_use)

        if data_config is None:
            data_config = SegmentationDataConfig(
                cache_path=cache_path,
                images_extractor=images_extractor,
                labels_extractor=labels_extractor,
            )

        dataset_output_mapper = DatasetOutputMapper(data_config=data_config)
        formatter = SegmentationBatchFormatter(
            data_config=data_config,
            class_names=class_names,
            class_names_to_use=class_names_to_use,
            n_image_channels=n_image_channels,
            threshold_value=threshold_soft_labels,
        )
        super().__init__(
            dataset_output_mapper=dataset_output_mapper,
            formatter=formatter,
            data_config=data_config,
            class_names=class_names,
        )
