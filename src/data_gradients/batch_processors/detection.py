from typing import List

from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.batch_processors.formatters.detection import DetectionBatchFormatter
from data_gradients.batch_processors.preprocessors.detection import DetectionBatchPreprocessor
from data_gradients.config.data.data_config import DetectionDataConfig


class DetectionBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        data_config: DetectionDataConfig,
        class_names: List[str],
        class_names_to_use: List[str],
        n_image_channels: int = 3,
    ):
        dataset_output_mapper = DatasetOutputMapper(data_config=data_config)
        formatter = DetectionBatchFormatter(
            data_config=data_config, class_names=class_names, class_names_to_use=class_names_to_use, n_image_channels=n_image_channels
        )
        preprocessor = DetectionBatchPreprocessor(class_names=class_names)

        super().__init__(dataset_output_mapper=dataset_output_mapper, batch_formatter=formatter, batch_preprocessor=preprocessor)
