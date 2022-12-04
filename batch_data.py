from dataclasses import dataclass
from typing import List


@dataclass
class BatchData:
    images: List
    labels: List
    onehot_labels: List
    onehot_contours: List
