from abc import ABC, abstractmethod

from batch_data import BatchData


class FeatureExtractorAbstract(ABC):

    def __init__(self, train_set: bool):
        self.train_set: bool = train_set

    @abstractmethod
    def execute(self, data: BatchData):
        pass

    @abstractmethod
    def process(self, ax):
        pass
