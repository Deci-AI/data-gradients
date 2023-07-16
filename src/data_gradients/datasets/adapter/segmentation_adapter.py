from typing import List, Optional, Iterable, Callable

import torch

from data_gradients.config.data.typing import SupportedDataType

from data_gradients.datasets.adapter.base_adapter import BaseDatasetAdapter
from data_gradients.batch_processors.adapters.dataset_adapter import DatasetAdapter
from data_gradients.batch_processors.preprocessors.segmentation import SegmentationBatchPreprocessor
from data_gradients.batch_processors.formatters.segmentation import SegmentationBatchFormatter
from data_gradients.config.data.data_config import SegmentationDataConfig
from data_gradients.utils.data_classes.data_samples import SegmentationSample


class SegmentationDatasetAdapter(BaseDatasetAdapter):
    """
    This is an abstract class that represents a dataset adapter.
    It acts as a bridge interface between any specific dataset/dataloader/raw data on disk and the analysis manager.
    """

    def __init__(
        self,
        data_iterable: Iterable,
        cache_filename: Optional[str] = None,
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
                cache_filename=cache_filename,
                images_extractor=images_extractor,
                labels_extractor=labels_extractor,
            )

        dataset_adapter = DatasetAdapter(data_config=data_config)
        formatter = SegmentationBatchFormatter(
            class_names=class_names,
            class_names_to_use=class_names_to_use,
            n_image_channels=n_image_channels,
            threshold_value=threshold_soft_labels,
        )
        super().__init__(data_iterable=data_iterable, dataset_adapter=dataset_adapter, formatter=formatter, data_config=data_config)

        self.preprocessor = SegmentationBatchPreprocessor(class_names=class_names)

    def samples_iterator(self, split_name: str) -> Iterable[SegmentationSample]:
        for (images, labels) in self:
            yield from self.preprocessor.preprocess(images, labels, split=split_name)
