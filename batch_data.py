from dataclasses import dataclass
from typing import List


@dataclass
class BatchData:
    images: List
    labels: List
    batch_onehot_contours: List
    batch_onehot_labels: List
