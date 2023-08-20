from abc import ABC, abstractmethod
from typing import Iterable

from data_gradients.utils.data_classes import ImageSample
from data_gradients.config.data.data_config import DataConfig
from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter


class BaseSampleIterable(ABC):
    """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing."""

    def __init__(self, dataset: BaseDatasetAdapter):
        self.dataset = dataset

    @abstractmethod
    def __iter__(self) -> Iterable[ImageSample]:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.
        :return:            Ready to analyse batch object, that depends on the current task (detection, segmentation, classification).
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def config(self) -> DataConfig:
        return self.dataset.data_config

    def close(self):
        self.config.close()
