from typing import List

from data_gradients.batch_processors.adapters.dataset_adapter import DatasetAdapter
from data_gradients.batch_processors.base import BatchProcessor
from data_gradients.batch_processors.formatters.classification import ClassificationBatchFormatter
from data_gradients.batch_processors.preprocessors.classification import ClassificationBatchPreprocessor
from data_gradients.config.data.data_config import ClassificationDataConfig


class ClassificationBatchProcessor(BatchProcessor):
    def __init__(
        self,
        *,
        data_config: ClassificationDataConfig,
        class_names: List[str],
        class_names_to_use: List[str],
        n_image_channels: int = 3,
    ):
        dataset_adapter = DatasetAdapter(data_config=data_config)
        formatter = ClassificationBatchFormatter(
            data_config=data_config, class_names=class_names, class_names_to_use=class_names_to_use, n_image_channels=n_image_channels
        )
        preprocessor = ClassificationBatchPreprocessor(class_names=class_names, n_image_channels=n_image_channels)

        super().__init__(dataset_adapter=dataset_adapter, batch_formatter=formatter, batch_preprocessor=preprocessor)
