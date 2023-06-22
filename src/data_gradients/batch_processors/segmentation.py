from typing import List

from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.adapters.dataset_adapter import DatasetAdapter
from data_gradients.batch_processors.preprocessors.segmentation import SegmentationBatchPreprocessor
from data_gradients.batch_processors.formatters.segmentation import SegmentationBatchFormatter
from data_gradients.config.data.data_config import SegmentationDataConfig


class SegmentationBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        data_config: SegmentationDataConfig,
        class_names: List[str],
        class_names_to_use: List[str],
        n_image_channels: int = 3,
        threshold_value: float = 0.5,
    ):

        dataset_adapter = DatasetAdapter(data_config=data_config)
        formatter = SegmentationBatchFormatter(
            class_names=class_names,
            class_names_to_use=class_names_to_use,
            n_image_channels=n_image_channels,
            threshold_value=threshold_value,
        )
        preprocessor = SegmentationBatchPreprocessor(class_names=class_names)

        super().__init__(dataset_adapter=dataset_adapter, batch_formatter=formatter, batch_preprocessor=preprocessor)
