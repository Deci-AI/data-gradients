from abc import abstractmethod

import numpy as np

from src.utils import SegBatchData
from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class SegmentationFeatureExtractorAbstract(FeatureExtractorAbstract):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def execute(self, data: SegBatchData):
        pass

    @abstractmethod
    def process(self, ax, train):
        pass

    @staticmethod
    def normalize_hist(hist):
        return list(np.array(hist) / sum(hist))
