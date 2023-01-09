from abc import ABC, abstractmethod

from PIL.Image import Image
import numpy as np
import torch
from torchvision.transforms import transforms

from src import preprocess
from src.utils import SegBatchData


class PreprocessorAbstract(ABC):

    def __init__(self, num_classes, images_extractor, labels_extractor, num_image_channels):
        self.number_of_classes: int = num_classes
        self._num_image_channels: int = num_image_channels
        self._container_mapper = {"first": None, "second": None}
        self._mappers = {'first': images_extractor, 'second': labels_extractor}

    @abstractmethod
    def validate(self, objects):
        pass

    @abstractmethod
    def preprocess(self, images, labels) -> SegBatchData:
        pass

    # TODO: Find a better way to pass route to logger

    @property
    def images_route(self):
        if self._container_mapper['first'] is not None:
            return {'get images': self._container_mapper['first'].route}
        else:
            return None

    @property
    def labels_route(self):
        if self._container_mapper['second'] is not None:
            return {'get labels': self._container_mapper['second'].route}
        else:
            return None

    @staticmethod
    def channels_last_to_first(tensors: torch.Tensor):
        """
        Permute BS, W, H, C -> BS, C, W, H
                0   1  2  3 -> 0   3  1  2
        :param tensors: Tensor[BS, W, H, C]
        :return: Tensor[BS, C, W, H]
        """
        return tensors.permute(0, 3, 1, 2)

    def _to_tensor(self, objs, tuple_place: str = "") -> torch.Tensor:
        """

        :param objs:
        :param tuple_place:
        :return:
        """
        if isinstance(objs, np.ndarray):
            return torch.from_numpy(objs)
        elif isinstance(objs, Image):
            return transforms.ToTensor()(objs)
        else:
            return self._handle_dict(objs, tuple_place)

    def _handle_dict(self, objs, tuple_place):
        """

        :param objs:
        :param tuple_place:
        :return:
        """
        if self._container_mapper[tuple_place] is not None:
            return self._container_mapper[tuple_place].container_to_tensor(objs)
        else:
            self._container_mapper[tuple_place] = preprocess.ContainerMapper()
            self._container_mapper[tuple_place].images = tuple_place == 'first'

            if self._mappers[tuple_place] is not None:
                self._container_mapper[tuple_place].mapper = self._mappers[tuple_place]
            else:
                self._container_mapper[tuple_place].get_mapping(objs)

            return self._container_mapper[tuple_place].container_to_tensor(objs)

