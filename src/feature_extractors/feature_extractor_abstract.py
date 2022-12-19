from abc import ABC, abstractmethod

from src.utils import SegBatchData


class FeatureExtractorAbstract(ABC):

    def __init__(self):
        self.colors = {0: 'red',
                       1: 'green'}
        self.single_axis: bool = True

    @abstractmethod
    def execute(self, data: SegBatchData):
        pass

    @abstractmethod
    def process(self, ax, train: bool):
        pass

    @staticmethod
    def normalize(values, total):
        return [((100 * value) / total) for value in values]
