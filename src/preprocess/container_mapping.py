from typing import Mapping, Callable, Optional, Any, List, Dict, Union
import json

from PIL import Image
import torch
from numpy import ndarray
from pygments import lexers, formatters, highlight
from torch import Tensor


class ContainerMapper:
    """
    Container Mapping class is supposed to analyze, track and extract images / labels data out of a python container,
    such as dictionary, json objects etc.
    It analyzes the type of the container and create a "route" to the data with using user's input.
    Then, it will create a mapper method that will apply on the container in order to extract the data.
    """
    def __init__(self):
        self._mapper: Optional[Callable] = None
        self._route: List[str] = []
        self._images: bool = True

    @property
    def images(self) -> bool:
        return self._images

    @images.setter
    def images(self, images: bool):
        self._images = images

    @property
    def route(self):
        return self._route

    def analyze(self, objs):
        """
        Check which kind of object we received. Raise a NotImplementedError exception if not handling the specific type.
        :param objs: any type of object, suppose to be a python container
        """
        if isinstance(objs, dict) or self.isjson(objs):
            self._get_dict_mapping(objs)
        else:
            raise NotImplementedError

    def _get_dict_mapping(self, objs: Union[Dict, str]):
        """
        Get either a Json serialized object or a dictionary object and find the "route" out of its keys.
        param objs: a Mapping type of object
        """
        self._route = self._get_users_string(objs, self.images)
        self._mapper = self._dict_mapping

    def _dict_mapping(self, objs):
        for r in self._route:
            objs = objs[r]
        return objs

    def container_to_tensor(self, objs) -> Tensor:
        return self._mapper(objs)

    @staticmethod
    def _get_users_string(objs, images):
        """
        Auxiliary method for the container_mapping recursive method. It holds the keys sequence target and ask the
        user to input which of the above keys mapping is the right one, in order to retrieve the correct data
        (either images or labels).
        :return: List of keys that if you will iterate with the Get Operation (d[k]) through all of them, you will get
                 the data you intended.
        """
        targets = []
        res = container_mapping(objs, path="", targets=targets)
        map_for_printing = json.dumps(res, indent=5, ensure_ascii=False)
        colorful_json = highlight(map_for_printing, lexers.JsonLexer(), formatters.TerminalFormatter())
        print(colorful_json.replace("\"", ""))
        value = int(input(f"Please insert the circled number of the required {'images' if images else 'labels'} data:\n"))
        print(f'Path for getting objects out of container: {targets[value]}')
        print('*' * 50)
        keys = [r.replace("'", "").replace('[', '').replace(']', '') for r in targets[value].split(']')][:-1]
        return keys

    @staticmethod
    def isjson(myjson) -> bool:
        """
        Method return if an object is a json serialized object or not
        :param myjson: any object
        :return: boolean if myjson is a json object
        """
        try:
            json.loads(myjson)
        except ValueError as e:
            return False
        else:
            return True


def container_mapping(obj: Any, path: str, targets: list):
    """
    Recursive function for "digging" into the mapping object it received and save a "path" to the target.
    Target is defined as one of [torch.Tensor, np.ndarray, PIL.Image],
    and if got Mapping / Sequence -> continue recursion.
    :param obj: recursively object returned
    :param path: current path - not achieved a target yet
    :param targets: current target's total path
    """
    if isinstance(obj, Mapping):
        printable_map = {}
        for k, v in obj.items():
            printable_map[k] = container_mapping(v, path + f"['{k}']", targets)
    elif isinstance(obj, tuple):
        types = []
        if len(obj) < 5:
            for o in obj:
                types.append(str(type(o)))
        else:
            types.append(str(type(obj[0])))
        printable_map = f' {numbers[len(targets) % len(numbers)]}: Tuple [{types}]'
        targets.append(path)
    elif isinstance(obj, list):
        printable_map = f' {numbers[len(targets) % len(numbers)]}: List [{type(obj[0])}]'
        targets.append(path)
    elif isinstance(obj, str):
        return "string"
        # targets.append(path)
    elif isinstance(obj, torch.Tensor):
        printable_map = f" {numbers[len(targets) % len(numbers)]}: Tensor"
        targets.append(path)
    elif isinstance(obj, ndarray):
        printable_map = f" {numbers[len(targets) % len(numbers)]}: ndarray"
        targets.append(path)
    elif isinstance(obj, Image.Image):
        printable_map = f" {numbers[len(targets) % len(numbers)]}: PIL Image"
        targets.append(path)
    else:
        raise RuntimeError(f"unsupported object! Object found has a type of {type(obj)} which is not supported for now.\n"
                           f"Supported types: [Mapping, Tuple, List, String, Tensor, Numpy array, PIL Image]")
    return printable_map


numbers = ["⓪", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
