from abc import ABC, abstractmethod
from typing import Iterable

from data_gradients.utils.data_classes import ImageSample
from data_gradients.config.data.data_config import DataConfig
from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter


class BaseSampleIterable(ABC):
    """
    Iterate over a dataset adapter and yield Sample objects one at a time.
    """

    def __init__(self, dataset: BaseDatasetAdapter):
        self.dataset = dataset

    @abstractmethod
    def __iter__(self) -> Iterable[ImageSample]:
        """
        :return:            Ready to analyse sample object, that depends on the current task (detection, segmentation, classification).
        """
        pass

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def config(self) -> DataConfig:
        return self.dataset.data_config

    def close(self):
        self.config.close()
