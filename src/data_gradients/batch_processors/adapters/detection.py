from data_gradients.config.data.data_config import DataConfig
from data_gradients.batch_processors.adapters.base import DatasetAdapter
from data_gradients.batch_processors.adapters.common_extractors.detection import CocoDetectionExtractor


class DetectionDatasetAdapter(DatasetAdapter):
    def __init__(self, data_config: DataConfig):
        super().__init__(data_config, predefined_extractors=[CocoDetectionExtractor()])
