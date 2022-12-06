from abc import ABC, abstractmethod

from utils.data_classes import BatchData
from preprocess.segmentation_preprocess import SegmentationPreprocessor


class PreprocessorAbstract(ABC):
    PREPROCESSORS = {'semantic-segmentation': SegmentationPreprocessor}

    def __init__(self):
        pass

    @staticmethod
    def get_preprocessor(task):
        return PreprocessorAbstract.PREPROCESSORS[task]()

    @abstractmethod
    def preprocess(self, images, labels) -> BatchData:
        pass
