import abc
import dataclasses
from typing import Optional, List, Iterable

import numpy as np


@dataclasses.dataclass
class SegmentationSample:
    """
    This is a dataclass that represents a single sample of the dataset.
    Support of different types of dataset formats is achieved by using adapters that should return SegmentationSample.

    Properties:
        sample_id: str
        image: np.ndarray of shape [H,W,C]
        mask: np.ndarray of shape [H,W] with integer values representing class labels
    """

    sample_id: str
    image: np.ndarray
    mask: np.ndarray


class SegmentationDatasetAdapter(abc.ABC):
    """
    This is an abstract class that represents a dataset adapter.
    It acts as a bridge interface between any specific dataset/dataloader/raw data on disk and the analysis manager.
    """

    @abc.abstractmethod
    def get_num_classes(self) -> int:
        ...

    @abc.abstractmethod
    def get_class_names(self) -> List[str]:
        ...

    @abc.abstractmethod
    def get_ignored_classes(self) -> Optional[List[int]]:
        ...

    @abc.abstractmethod
    def get_iterator(self) -> Iterable[SegmentationSample]:
        """
        This method should return an iterable object that contains SegmentationSample objects.
        It could read samples from disk, existing dataset, dataloader, etc.
        """
        raise NotImplementedError()

    def __len__(self):
        return None
