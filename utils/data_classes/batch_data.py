from dataclasses import dataclass
from typing import List

from torch import Tensor


@dataclass
class BatchData:
    images: Tensor
    labels: Tensor
    batch_onehot_contours: List
    batch_onehot_labels: List
