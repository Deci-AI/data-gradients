from typing import Tuple

from torch import Tensor

from data_gradients.batch_processors.validators.base import BatchValidator


class DetectionBatchValidator(BatchValidator):
    def __init__(self):
        pass

    def __call__(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        return images, labels
