from typing import Optional, Callable

from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.extractors.batch_extractor import BatchExtractor
from data_gradients.batch_processors.validators.detection import DetectionBatchValidator
from data_gradients.batch_processors.preprocessors.detection import DetectionBatchPreprocessor


class SegmentationBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
    ):
        extractor = BatchExtractor(
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
        )
        validator = DetectionBatchValidator()
        preprocessor = DetectionBatchPreprocessor()

        super().__init__(batch_extractor=extractor, batch_validator=validator, batch_preprocessor=preprocessor)
