from typing import List, Optional, Callable

from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.adapters.dataset_adapter import DatasetAdapter
from data_gradients.batch_processors.preprocessors.segmentation import SegmentationBatchPreprocessor
from data_gradients.batch_processors.formatters.segmentation import SegmentationBatchFormatter


class SegmentationBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        class_names: List[str],
        n_image_channels: int = 3,
        threshold_value: float = 0.5,
        ignore_labels: Optional[List[int]] = None,
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
    ):

        dataset_adapter = DatasetAdapter(
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
        )
        formatter = SegmentationBatchFormatter(
            n_classes=len(class_names),
            n_image_channels=n_image_channels,
            threshold_value=threshold_value,
            ignore_labels=ignore_labels,
        )
        preprocessor = SegmentationBatchPreprocessor(class_names=class_names)

        super().__init__(dataset_adapter=dataset_adapter, batch_formatter=formatter, batch_preprocessor=preprocessor)
