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
    def _execute(self, data: SegBatchData):
        pass

    @abstractmethod
    def _post_process(self, split: str):
        pass

    @staticmethod
    def normalize(values, total):
        if total == 0:
            total = 1
        return [np.round(((100 * value) / total), 3) for value in values]
