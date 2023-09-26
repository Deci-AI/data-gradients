from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.dataset_adapters.config.data_config import ClassificationDataConfig
from data_gradients.dataset_adapters.formatters.classification import ClassificationBatchFormatter


class ClassificationDatasetAdapter(BaseDatasetAdapter):
    """Wrap a classification dataset so that it would return standardized tensors.

    :param n_image_channels:    Number of image channels.
    :param data_config:         Instance of DetectionDataConfig class that manages dataset/dataloader configurations.
    """

    def __init__(self, data_config: ClassificationDataConfig, n_image_channels: int = 3):
        dataset_output_mapper = DatasetOutputMapper(data_config=data_config)
        formatter = ClassificationBatchFormatter(data_config=data_config, n_image_channels=n_image_channels)
        super().__init__(dataset_output_mapper=dataset_output_mapper, formatter=formatter, data_config=data_config)

    @classmethod
    def from_cache(cls, cache_path: str) -> "ClassificationDatasetAdapter":
        return cls(data_config=ClassificationDataConfig(cache_path=cache_path))
