from typing import Tuple

from torch import Tensor

from data_gradients.batch_processors.formatters.base import BatchFormatter


class DetectionBatchFormatter(BatchFormatter):
    def __init__(self):
        pass

    def format(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        return images, labels
