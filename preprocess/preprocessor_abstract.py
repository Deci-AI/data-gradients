from abc import ABC, abstractmethod

from PIL.Image import Image
import numpy as np
from numpy import ndarray
import torch
from torchvision.transforms import transforms

import preprocess
from utils.data_classes import BatchData
import json
from pygments import highlight, lexers, formatters
from typing import Mapping, Sequence, Any


class PreprocessorAbstract(ABC):

    def __init__(self):
        self.number_of_classes: int = 0
        self._number_of_channels: int = 3
        self._is_container: bool = False

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

    def _to_tensor(self, objs) -> torch.Tensor:
        if isinstance(objs, np.ndarray):
            return torch.from_numpy(objs)
        elif isinstance(objs, Image):
            return transforms.ToTensor()(objs)
        elif self._is_container:
            return self._container_to_tensor(objs)
        else:
            self._analyze_container(objs)

    def _analyze_container(self, objs):
        raise NotImplementedError
        sample = {
            "input": [torch.ones(1, 1, 1), ndarray(shape=(1, 1, 1))],
            "target": {"attributes": ndarray(shape=(1, 1, 1)), "mask": ndarray(shape=(1, 1, 1)),
                       "labels": ndarray(shape=(1, 1, 1))},
        }
        print("Dataset analyzer has detected an unknown container format from you dataset __get_item__() function:")
        print(f'{type(objs)}')
        targets = []
        res = container_mapping(sample, path="", targets=targets)
        map_for_printing = json.dumps(res, indent=5, ensure_ascii=False)
        colorful_json = highlight(map_for_printing, lexers.JsonLexer(), formatters.TerminalFormatter())
        print(colorful_json.replace("\"", ""))
        value = input("which one of the yellow items is your Input image?\n")
        print(f'path to Input image: sample{targets[int(value)]}')
        value = input("which one of the yellow items is a mask containing the segments?\n")
        print(f'path to Segments mask: sample{targets[int(value)]}')

    def _container_to_tensor(self, objs):
        pass
        return objs

def container_mapping(obj: Any, path: str, targets: list):
    if isinstance(obj, Mapping):
        printable_map = {}
        for k, v in obj.items():
            printable_map[k] = container_mapping(v, path + f"['{k}']", targets)
    elif isinstance(obj, Sequence):
        printable_map = []
        for i, o in enumerate(obj):
            printable_map.append(f'{i}: ' + container_mapping(o, path + f"[{i}]", targets))
    elif isinstance(obj, torch.Tensor):
        printable_map = f"Tensor {numbers[len(targets)]}"
        targets.append(path)
    elif isinstance(obj, ndarray):
        printable_map = f"ndarray {numbers[len(targets)]}"
        targets.append(path)
    # TODO add also PIL.Image
    else:
        raise RuntimeError("unsupported object")

    return printable_map


numbers = [
    "⓪",
    "①",
    "②",
    "③",
    "④",
    "⑤",
    "⑥",
    "⑦",
    "⑧",
    "⑨",
    "⑩",
]
