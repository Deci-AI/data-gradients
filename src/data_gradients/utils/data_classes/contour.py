from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass()
class Contour:
    points: np.array
    area: float
    w: float
    h: float
    center: Tuple[int, int]
    perimeter: float
    class_id: int
    bbox_area: float
