from abc import abstractmethod

import numpy as np

from utils.data_classes import BatchData
from feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class SegmentationFeatureExtractorAbstract(FeatureExtractorAbstract):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def execute(self, data: BatchData):
        pass

    @abstractmethod
    def process(self, ax, train):
        pass

    @staticmethod
    def normalize_hist(hist):
        return list(np.array(hist) / sum(hist))
