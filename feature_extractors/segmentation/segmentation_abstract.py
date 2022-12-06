from abc import abstractmethod

from batch_data import BatchData
from feature_extractors import FeatureExtractorBuilder


class SegmentationFeatureExtractorAbstract(FeatureExtractorBuilder):

    def __init__(self, train_set: bool):
        super().__init__(train_set)

    @abstractmethod
    def execute(self, data: BatchData):
        pass

    @abstractmethod
    def process(self, ax):
        pass
