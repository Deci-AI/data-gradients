from abc import ABC, abstractmethod

from src.utils import BatchData


class FeatureExtractorAbstract(ABC):

    def __init__(self):
        self.colors = {0: 'red',
                       1: 'green'}
        self.single_axis: bool = True

    @abstractmethod
    def execute(self, data: BatchData):
        pass

    @abstractmethod
    def process(self, ax, train: bool):
        pass
