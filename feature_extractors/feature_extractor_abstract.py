from abc import ABC, abstractmethod

from utils.data_classes import BatchData


class FeatureExtractorAbstract(ABC):

    def __init__(self):
        self.colors = {0: 'red',
                       1: 'green'}

    @abstractmethod
    def execute(self, data: BatchData):
        pass

    @abstractmethod
    def process(self, ax, train: bool):
        pass
