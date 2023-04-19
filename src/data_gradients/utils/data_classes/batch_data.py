from dataclasses import dataclass
from typing import List

from torch import Tensor

from data_gradients.utils.data_classes.contour import Contour


@dataclass()
class BatchData:
    """
    Images - [BS, 3, W, H]
    Labels - [BS, N, W, H] where N is number of classes in each image
    split  - train / val
    """

    images: Tensor
    labels: Tensor
    split: str


@dataclass
class SegBatchData(BatchData):
    """
    contours - [BS, N, C, P, 1, 2] where (P, 1, 2) is a contour representation, C is number of contours and N
                                   is number of classes
    """

    contours: List[List[List[Contour]]]


@dataclass
class Sample:
    """
    :attr images:  [3, W, H]
    :attr labels:  [N, W, H] where N is number of classes in each image
    :attr split:   train / val
    """

    images: Tensor
    labels: Tensor
    split: str


@dataclass
class SegmentationSample(Sample):
    """
    :attr contours: [N, C, P, 1, 2] where (P, 1, 2) is a contour representation, C is number of contours and N is number of classes
    """

    contours: List[List[Contour]]
