from typing import List, Optional, Callable

import torch

from data_gradients.config.data.typing import SupportedDataType
from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.dataset_adapters.formatters.detection import DetectionBatchFormatter
from data_gradients.config.data.data_config import DetectionDataConfig


class DetectionDatasetAdapter(BaseDatasetAdapter):
    """Wrap a detection dataset so that it would return standardized tensors.

    :param cache_path:          The filename of the cache file.
    :param n_classes:           The number of classes.
    :param class_names:         List of class names.
    :param class_names_to_use:  List of class names to use.
    :param images_extractor:    Callable function for extracting images.
    :param labels_extractor:    Callable function for extracting labels.
    :param is_label_first:      A flag to indicate if labels are the first entity in the dataset.
    :param bbox_format:         Callable function for formatting bounding boxes.
    :param n_image_channels:    Number of image channels.
    :param data_config:         Instance of DetectionDataConfig class that manages dataset/dataloader configurations.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        n_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        class_names_to_use: Optional[List[str]] = None,
        images_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]] = None,
        labels_extractor: Optional[Callable[[SupportedDataType], torch.Tensor]] = None,
        is_label_first: Optional[bool] = None,
        bbox_format: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_image_channels: int = 3,
        data_config: Optional[DetectionDataConfig] = None,
    ):
        class_names = self.resolve_class_names(class_names=class_names, n_classes=n_classes)
        class_names_to_use = self.resolve_class_names_to_use(class_names=class_names, class_names_to_use=class_names_to_use)

        if data_config is None:
            data_config = DetectionDataConfig(
                cache_path=cache_path,
                images_extractor=images_extractor,
                labels_extractor=labels_extractor,
                is_label_first=is_label_first,
                xyxy_converter=bbox_format,
            )

        dataset_output_mapper = DatasetOutputMapper(data_config=data_config)
        formatter = DetectionBatchFormatter(
            data_config=data_config,
            class_names=class_names,
            class_names_to_use=class_names_to_use,
            n_image_channels=n_image_channels,
        )
        super().__init__(
            dataset_output_mapper=dataset_output_mapper,
            formatter=formatter,
            data_config=data_config,
            class_names=class_names,
        )
