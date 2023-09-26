from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.dataset_adapters.formatters.segmentation import SegmentationBatchFormatter
from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig


class SegmentationDatasetAdapter(BaseDatasetAdapter):
    """Wrap a segmentation dataset so that it would return standardized tensors.

    :param threshold_soft_labels:   Soft labels threshold.
    :param data_config:             Instance of DetectionDataConfig class that manages dataset/dataloader configurations.
    """

    def __init__(self, data_config: SegmentationDataConfig, threshold_soft_labels: float = 0.5):
        dataset_output_mapper = DatasetOutputMapper(data_config=data_config)
        formatter = SegmentationBatchFormatter(data_config=data_config, threshold_value=threshold_soft_labels)
        super().__init__(dataset_output_mapper=dataset_output_mapper, formatter=formatter, data_config=data_config)

    @classmethod
    def from_cache(cls, cache_path: str) -> "SegmentationDatasetAdapter":
        return cls(data_config=SegmentationDataConfig(cache_path=cache_path))
