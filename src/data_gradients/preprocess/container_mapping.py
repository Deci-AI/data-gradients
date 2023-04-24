from typing import Mapping, Optional, Any, List
import json

from PIL import Image
import torch
from numpy import ndarray
from pygments import lexers, formatters, highlight
from torch import Tensor


class TensorExtractor:
    """
    DataPathFinder analyzes, tracks, and extracts image/label data from a Python container,
    such as a dictionary or JSON objects. It determines the container type and creates a "data_path" to the
    data using user input. Then, it generates a mapping method that will be applied to the container
    in order to extract the data.
    """

    def __init__(self, search_for_images: bool, objs: Any):
        self.path_to_tensor: Optional[List[str]] = prompt_user_for_data_keys(objs=objs, search_for_images=search_for_images)

    def __call__(self, objs: Any) -> Tensor:
        return traverse_nested_data_structure(data=objs, keys=self.path_to_tensor)


def prompt_user_for_data_keys(objs: Any, search_for_images: bool) -> List[str]:
    """
    Auxiliary method for the container_mapping recursive method. It holds the keys sequence target and asks the
    user to input which of the above keys mapping is the right one in order to retrieve the correct data
    (either images or labels).

    :return: List of keys that if you iterate with the Get Operation (d[k]) through all of them, you will get
             the data you intended.
    """

    if not (isinstance(objs, dict) or is_valid_json(objs)):
        raise NotImplementedError(f"type{type(objs)} not currently supported")  # TODO raise custom exception and catch it later on

    targets = []
    mapping = objects_mapping(objs, path="", targets=targets)

    mapping_str = json.dumps(mapping, indent=4, ensure_ascii=False)
    mapping_str = highlight(mapping_str, lexers.JsonLexer(), formatters.TerminalFormatter())
    mapping_str = mapping_str.replace('"', "")
    print(mapping_str)

    value = int(input(f"Please insert the circled number of the required {'images' if search_for_images else 'labels'} data:\n"))
    print(f"Path for getting objects out of container: {targets[value]}")
    print("************************************************************")

    keys = [r.replace("'", "").replace("[", "").replace("]", "") for r in targets[value].split("]")][:-1]
    return keys


def is_valid_json(myjson: str) -> bool:
    """Check if an object is a JSON serialized object or not.

    :param myjson: any object
    :return: boolean if myjson is a JSON object
    """
    try:
        json.loads(myjson)
    except ValueError:
        return False
    else:
        return True


def objects_mapping(obj: Any, path: str, targets: List[str]) -> Any:
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
            printable_map[k] = objects_mapping(v, path + f"['{k}']", targets)
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
