from typing import List, Optional, Callable

from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.extractors.batch_extractor import BatchExtractor
from data_gradients.batch_processors.preprocessors.segmentation import SegmentationBatchPreprocessor
from data_gradients.batch_processors.formatters.segmentation import SegmentationBatchFormatter


class SegmentationBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        n_classes: int,
        n_image_channels: int,
        threshold_value: float,
        ignore_labels: Optional[List[int]] = None,
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
    ):
        extractor = BatchExtractor(
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
        )
        formatter = SegmentationBatchFormatter(
            n_classes=n_classes,
            n_image_channels=n_image_channels,
            threshold_value=threshold_value,
            ignore_labels=ignore_labels,
        )
        preprocessor = SegmentationBatchPreprocessor()

        super().__init__(batch_extractor=extractor, batch_formatter=formatter, batch_preprocessor=preprocessor)
