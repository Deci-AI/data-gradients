from typing import Optional, Callable, List

from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.adapters.dataset_adapter import DatasetAdapter
from data_gradients.batch_processors.formatters.detection import DetectionBatchFormatter
from data_gradients.batch_processors.preprocessors.detection import DetectionBatchPreprocessor


class DetectionBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        class_names: List[str],
        class_names_to_use: List[str],
        n_image_channels: int = 3,
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
    ):
        dataset_adapter = DatasetAdapter(
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
        )
        formatter = DetectionBatchFormatter(class_names=class_names, class_names_to_use=class_names_to_use, n_image_channels=n_image_channels)
        preprocessor = DetectionBatchPreprocessor(class_names=class_names)

        super().__init__(dataset_adapter=dataset_adapter, batch_formatter=formatter, batch_preprocessor=preprocessor)
