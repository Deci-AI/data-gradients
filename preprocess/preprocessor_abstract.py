from abc import ABC, abstractmethod

from PIL.Image import Image
import numpy as np
import torch
from torchvision.transforms import transforms

import preprocess
from utils.data_classes import BatchData


class PreprocessorAbstract(ABC):

    def __init__(self):
        self.number_of_classes: int = 0
        self._number_of_channels: int = 3

    @staticmethod
    def get_preprocessor(task):
        return preprocess.PREPROCESSORS[task]()

    @abstractmethod
    def validate(self, images, labels):
        pass

    @abstractmethod
    def preprocess(self, images, labels) -> BatchData:
        pass

    @staticmethod
    def channels_last_to_first(tensors: torch.Tensor):
        return tensors.permute(0, 3, 1, 2)

    @staticmethod
    def _to_tensor(objs) -> torch.Tensor:
        if isinstance(objs, np.ndarray):
            return torch.from_numpy(objs)
        elif isinstance(objs, Image):
            return transforms.ToTensor()(objs)
