from abc import ABC, abstractmethod
from typing import Mapping, Optional, Union, Callable, Any, List

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms

from data_gradients.preprocess import TensorFinder
from data_gradients.utils import BatchData


class PreprocessorAbstract(ABC):
    def __init__(
        self,
        num_classes: int,
        num_image_channels: int,
        images_extractor: Optional[Callable] = None,  # TODO: add more typing
        labels_extractor: Optional[Callable] = None,
    ):
        self.number_of_classes = num_classes
        self._num_image_channels = num_image_channels
        self._tensor_finders = {0: images_extractor, 1: labels_extractor}

    @abstractmethod
    def validate(self, objects):
        pass

    @abstractmethod
    def preprocess(self, images: torch.Tensor, labels) -> BatchData:
        pass

    @property
    def images_route(self) -> List[str]:
        tensor_finder = self._tensor_finders[0]
        # TODO: Check removing the previous dict structure will not lead to backward compatibility
        return tensor_finder.path_to_object if isinstance(tensor_finder, TensorFinder) else []

    @property
    def labels_route(self) -> List[str]:
        tensor_finder = self._tensor_finders[1]
        return tensor_finder.path_to_object if isinstance(tensor_finder, TensorFinder) else []

    def _to_tensor(self, objs: Union[np.ndarray, PIL.Image.Image, Mapping], tuple_idx: int) -> torch.Tensor:
        if isinstance(objs, np.ndarray):
            return torch.from_numpy(objs)
        elif isinstance(objs, PIL.Image.Image):
            return transforms.ToTensor()(objs)
        else:
            return self.extract_tensor_from_complex_data(objs=objs, tuple_idx=tuple_idx)

    def extract_tensor_from_complex_data(self, objs: Any, tuple_idx: int) -> torch.Tensor:
        mapping_fn = self.get_tensor_finder(tuple_idx=tuple_idx, objs=objs)
        return mapping_fn(objs)

    def get_tensor_finder(self, objs: Any, tuple_idx: int) -> Union[Callable, TensorFinder]:
        if self._tensor_finders[tuple_idx] is None:
            self._tensor_finders[tuple_idx] = TensorFinder(search_for_images=(tuple_idx == 0), objs=objs)
        return self._tensor_finders[tuple_idx]


class BatchValidatorAbstract(ABC):
    def __init__(
        self,
        num_classes: int,
        num_image_channels: int,
        images_extractor: Optional[Callable] = None,  # TODO: add more typing
        labels_extractor: Optional[Callable] = None,
    ):
        self.number_of_classes = num_classes
        self._num_image_channels = num_image_channels
        self._tensor_finders = {0: images_extractor, 1: labels_extractor}

    @abstractmethod
    def validate(self, objects):
        pass

    @abstractmethod
    def preprocess(self, images: torch.Tensor, labels) -> BatchData:
        pass

    @property
    def images_route(self) -> List[str]:
        tensor_finder = self._tensor_finders[0]
        # TODO: Check removing the previous dict structure will not lead to backward compatibility
        return tensor_finder.path_to_object if isinstance(tensor_finder, TensorFinder) else []

    @property
    def labels_route(self) -> List[str]:
        tensor_finder = self._tensor_finders[1]
        return tensor_finder.path_to_object if isinstance(tensor_finder, TensorFinder) else []

    def _to_tensor(self, objs: Union[np.ndarray, PIL.Image.Image, Mapping], tuple_idx: int) -> torch.Tensor:
        if isinstance(objs, np.ndarray):
            return torch.from_numpy(objs)
        elif isinstance(objs, PIL.Image.Image):
            return transforms.ToTensor()(objs)
        else:
            return self.extract_tensor_from_complex_data(objs=objs, tuple_idx=tuple_idx)

    def extract_tensor_from_complex_data(self, objs: Any, tuple_idx: int) -> torch.Tensor:
        mapping_fn = self.get_tensor_finder(tuple_idx=tuple_idx, objs=objs)
        return mapping_fn(objs)

    def get_tensor_finder(self, objs: Any, tuple_idx: int) -> Union[Callable, TensorFinder]:
        if self._tensor_finders[tuple_idx] is None:
            self._tensor_finders[tuple_idx] = TensorFinder(search_for_images=(tuple_idx == 0), objs=objs)
        return self._tensor_finders[tuple_idx]
