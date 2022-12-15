from abc import ABC, abstractmethod

from PIL.Image import Image
import numpy as np
import torch
from torchvision.transforms import transforms

from src import preprocess
from src.utils import BatchData


class PreprocessorAbstract(ABC):

    def __init__(self, num_classes=0):
        self.number_of_classes: int = num_classes
        self._number_of_channels: int = 3
        self._container_mapper = {"first": None, "second": None}

    @abstractmethod
    def validate(self, objects):
        pass

    @abstractmethod
    def preprocess(self, images, labels) -> BatchData:
        pass

    @staticmethod
    def channels_last_to_first(tensors: torch.Tensor):
        return tensors.permute(0, 3, 1, 2)

    def _to_tensor(self, objs, tuple_place: str = "") -> torch.Tensor:
        if isinstance(objs, np.ndarray):
            return torch.from_numpy(objs)
        elif isinstance(objs, Image):
            return transforms.ToTensor()(objs)
        elif self._container_mapper[tuple_place] is not None:
            return self._container_mapper[tuple_place].container_to_tensor(objs)
        else:
            self._container_mapper[tuple_place] = preprocess.ContainerMapper()
            self._container_mapper[tuple_place].analyze(objs)
            return self._container_mapper[tuple_place].container_to_tensor(objs)
