from abc import abstractmethod

import numpy as np

from src.utils import SegBatchData
from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class SegmentationFeatureExtractorAbstract(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor abstract class.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def execute(self, data: SegBatchData):
        pass

    @abstractmethod
    def process(self, ax, train):
        pass

    @staticmethod
    def normalize(values, total):
        return [np.round(((100 * value) / total), 3) for value in values]
