from typing import Mapping, Callable, Optional, Any, List, Dict, Union
import json

from PIL import Image
import torch
from numpy import ndarray
from pygments import lexers, formatters, highlight
from torch import Tensor


class DataPathFinder:
    """
    DataPathFinder analyzes, tracks, and extracts image/label data from a Python container,
    such as a dictionary or JSON objects. It determines the container type and creates a "data_path" to the
    data using user input. Then, it generates a mapping method that will be applied to the container
    in order to extract the data.
    """

    def __init__(self):
        self.mapper: Optional[Callable] = None
        self._data_path: List[str] = []
        self.search_for_images: bool = True

    @property
    def data_path(self) -> List[str]:
        return self._data_path

    def determine_mapping(self, objs: Any):
        """
        Check the type of the input object. Raise a NotImplementedError exception if the specific type is not handled.

        :param objs: any type of object, supposed to be a Python container
        """
        if isinstance(objs, dict) or self.is_json(objs):
            self._find_dict_path(objs)
        else:
            raise NotImplementedError

    def _find_dict_path(self, objs: Union[Dict, str]):
        """
        Process either a JSON serialized object or a dictionary object and find the "data_path" using its keys.

        :param objs: a Mapping type of object
        """
        self._data_path = self._get_user_input(objs, self.search_for_images)
        self.mapper = traverse_nested_data_structure

    def container_to_tensor(self, objs: Any) -> Tensor:
        return self.mapper(objs)

    @staticmethod
    def _get_user_input(objs: Any, search_for_images: bool) -> List[str]:
        """
        Auxiliary method for the container_mapping recursive method. It holds the keys sequence target and asks the
        user to input which of the above keys mapping is the right one in order to retrieve the correct data
        (either images or labels).

        :return: List of keys that if you iterate with the Get Operation (d[k]) through all of them, you will get
                 the data you intended.
        """
        targets = []
        res = container_mapping(objs, path="", targets=targets)
        map_for_printing = json.dumps(res, indent=4, ensure_ascii=False)
        colorful_json = highlight(map_for_printing, lexers.JsonLexer(), formatters.TerminalFormatter())
        print(colorful_json.replace('"', ""))
        value = int(input(f"Please insert the circled number of the required {'images' if search_for_images else 'labels'} data:\n"))
        print(f"Path for getting objects out of container: {targets[value]}")
        print("*" * 50)
        keys = [r.replace("'", "").replace("[", "").replace("]", "") for r in targets[value].split("]")][:-1]
        return keys

    @staticmethod
    def is_json(myjson) -> bool:
        """
        Method returns if an object is a JSON serialized object or not.

        :param myjson: any object
        :return: boolean if myjson is a JSON object
        """
        try:
            json.loads(myjson)
        except ValueError:
            return False
        else:
            return True


def container_mapping(obj: Any, path: str, targets: List[str]) -> Any:
    """
    Recursive function for "digging" into the mapping object it received and save a "path" to the target.
    Target is defined as one of [torch.Tensor, np.ndarray, PIL.Image],
    and if got Mapping / Sequence -> continue recursion.

    :param obj: recursively returned object
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
        printable_map = f" {numbers[len(targets) % len(numbers)]}: Tuple [{types}]"
        targets.append(path)
    elif isinstance(obj, list):
        printable_map = f" {numbers[len(targets) % len(numbers)]}: List [{type(obj[0])}]"
        targets.append(path)
    elif isinstance(obj, str):
        return "string"
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
        raise RuntimeError(
            f"Unsupported object! Object found has a type of {type(obj)} which is not supported for now.\n"
            f"Supported types: [Mapping, Tuple, List, String, Tensor, Numpy array, PIL Image]"
        )
    return printable_map


def traverse_nested_data_structure(data: Mapping, keys: List[str]) -> Any:
    """Traverse a nested data structure and returns the value at the specified key path.

    :param data:    Nested data structure like dict, defaultdict or OrderedDict
    :param keys:    List of strings representing the keys in the data structure
    :return:        Value at the specified key path in the data structure
    """
    for key in keys:
        data = data[key]
    return data


numbers = ["⓪", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
