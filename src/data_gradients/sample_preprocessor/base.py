from abc import ABC, abstractmethod
from typing import Iterator, Iterable

from data_gradients.utils.data_classes import ImageSample
from data_gradients.config.data.typing import SupportedDataType
from data_gradients.config.data.data_config import DataConfig


class BaseSamplePreprocessor(ABC):
    """
    Iterate over a dataset adapter and yield Sample objects one at a time.
    """

    def __init__(self, config: DataConfig):
        self.config = config

    def close(self):
        self.config.close()

    @abstractmethod
    def preprocess_samples(self, dataset: Iterable[SupportedDataType], split: str) -> Iterator[ImageSample]:
        ...
