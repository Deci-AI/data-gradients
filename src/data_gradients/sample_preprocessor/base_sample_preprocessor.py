from abc import ABC, abstractmethod
from typing import Iterator, Iterable

from data_gradients.utils.data_classes import ImageSample
from data_gradients.dataset_adapters.config.typing import SupportedDataType
from data_gradients.dataset_adapters.config.data_config import DataConfig


class BaseSamplePreprocessor(ABC):
    def __init__(self, config: DataConfig):
        self.config = config

    @abstractmethod
    def preprocess_samples(self, dataset: Iterable[SupportedDataType], split: str) -> Iterator[ImageSample]:
        ...
