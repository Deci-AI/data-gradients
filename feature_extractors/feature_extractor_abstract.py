from abc import ABC, abstractmethod
from typing import Dict

from batch_data import BatchData


class FeatureExtractorBuilder(ABC):

    def __init__(self, train_set: bool):
        self.train_set: bool = train_set

    @abstractmethod
    def execute(self, data: BatchData):
        pass

    @abstractmethod
    def process(self, ax):
        pass
