from dataclasses import dataclass
from typing import List

from torch import Tensor


@dataclass()
class BatchData:
    """
        Images - [BS, 3, W, H]
        Labels - [BS, N, W, H] where N is number of classes in each image
    """
    images: Tensor
    labels: List


@dataclass
class SegBatchData(BatchData):
    """
    contours - [BS, N, C, P, 1, 2] where (P, 1, 2) is a contour representation, C is number of contours and N
                                   is number of classes
    """
    contours: List
