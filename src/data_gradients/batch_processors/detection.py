from typing import Optional, Callable

from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.extractors.batch_extractor import BatchExtractor
from data_gradients.batch_processors.formatters.detection import DetectionBatchFormatter
from data_gradients.batch_processors.preprocessors.detection import DetectionBatchPreprocessor


class DetectionBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        n_image_channels: int,
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
    ):
        extractor = BatchExtractor(
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
        )
        formatter = DetectionBatchFormatter(n_image_channels=n_image_channels)
        preprocessor = DetectionBatchPreprocessor()

        super().__init__(batch_extractor=extractor, batch_formatter=formatter, batch_preprocessor=preprocessor)
