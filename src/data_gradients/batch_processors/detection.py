from typing import List, Optional, Iterable, Callable

import torch

from data_gradients.config.data.typing import SupportedDataType
from data_gradients.utils.data_classes.data_samples import DetectionSample
from data_gradients.batch_processors.base import BaseDatasetAdapter
from data_gradients.batch_processors.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.batch_processors.formatters.detection import DetectionBatchFormatter
from data_gradients.batch_processors.preprocessors.detection import DetectionBatchPreprocessor
from data_gradients.config.data.data_config import DetectionDataConfig


class DetectionDatasetAdapter(BaseDatasetAdapter):
    def __init__(
        self,
        data_iterable: Iterable,
        cache_filename: Optional[str] = None,
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
        self.data_iterable = data_iterable

        class_names = self.resolve_class_names(class_names=class_names, n_classes=n_classes)
        class_names_to_use = self.resolve_class_names_to_use(class_names=class_names, class_names_to_use=class_names_to_use)

        if data_config is None:
            data_config = DetectionDataConfig(
                cache_filename=cache_filename,
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
        super().__init__(data_iterable=data_iterable, dataset_output_mapper=dataset_output_mapper, formatter=formatter, data_config=data_config)

        self.preprocessor = DetectionBatchPreprocessor(class_names=class_names)

    def samples_iterator(self, split_name: str) -> Iterable[DetectionSample]:
        for (images, labels) in iter(self):
            yield from self.preprocessor.preprocess(images, labels, split=split_name)
