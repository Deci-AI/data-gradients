from abc import ABC, abstractmethod

import torch

from data_gradients.utils import BatchData


class Preprocessor(ABC):
    @abstractmethod
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> BatchData:
        pass
