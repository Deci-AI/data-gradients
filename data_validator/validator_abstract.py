from abc import abstractmethod
from typing import Iterator

from data_validator.segmentation_validator import SegmentationValidator


class ValidatorAbstract:
    VALIDATORS = {'semantic-segmentation': SegmentationValidator}

    def __init__(self):
        self._number_of_channels = 3

    @staticmethod
    def get_validator(task):
        return ValidatorAbstract.VALIDATORS[task]()

    @abstractmethod
    def validate(self, data_iterator: Iterator):
        pass

