from data_gradients.config.data.data_config import SegmentationDataConfig
from data_gradients.batch_processors.adapters.base import DatasetAdapter


class SegmentationDatasetAdapter(DatasetAdapter):
    def __init__(self, data_config: SegmentationDataConfig):
        super().__init__(data_config, predefined_extractors=[])
