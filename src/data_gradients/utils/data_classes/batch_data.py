from dataclasses import dataclass
from typing import List
from abc import ABC

from torch import Tensor

from data_gradients.utils.data_classes.contour import Contour


@dataclass
class BatchData(ABC):
    """
    Images - [BS, 3, W, H]
    Labels - [BS, N, W, H] where N is number of classes in each image
    split  - train / val
    """

    split: str
    images: Tensor


@dataclass
class SegmentationBatchData(BatchData):
    """
    contours - [BS, N, C, P, 1, 2] where (P, 1, 2) is a contour representation, C is number of contours and N
                                   is number of classes
    """

    labels: Tensor
    contours: List[List[List[Contour]]]


@dataclass
class DetectionBatchData(BatchData):
    labels: Tensor
    bboxes: Tensor
