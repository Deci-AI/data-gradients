from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BatchValidator(ABC):
    @abstractmethod
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
