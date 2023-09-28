from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.dataset_adapters.config.data_config import ClassificationDataConfig
from data_gradients.dataset_adapters.formatters.classification import ClassificationBatchFormatter


class ClassificationDatasetAdapter(BaseDatasetAdapter):
    """Wrap a classification dataset so that it would return standardized tensors.

    :param data_config:         Instance of DetectionDataConfig class that manages dataset/dataloader configurations.
    """

    def __init__(self, data_config: ClassificationDataConfig):
        dataset_output_mapper = DatasetOutputMapper(data_config=data_config)
        formatter = ClassificationBatchFormatter(data_config=data_config)
        super().__init__(dataset_output_mapper=dataset_output_mapper, formatter=formatter, data_config=data_config)

    @classmethod
    def from_cache(cls, cache_path: str) -> "ClassificationDatasetAdapter":
        return cls(data_config=ClassificationDataConfig(cache_path=cache_path))
