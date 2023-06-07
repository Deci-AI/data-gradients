from typing import Mapping, Callable, Dict, Any, List, Tuple, Sequence, Union
import re

from PIL import Image
import torch
from numpy import ndarray


def get_tensor_extractor_options(objs: Any) -> Dict[str, Callable]:
    """Extract out of objs all the potential fields of type [torch.Tensor, np.ndarray, PIL.Image], and then
    asks the user to input which of the above keys mapping is the right one in order to retrieve the correct data (either images or labels).

    :param objs:        Dictionary of json-like structure.
    """

    paths = []
    objects_mapping(objs, path="", targets=paths)

    options = {}
    for k, v in paths:
        option_key = f"{k}: {v}"
        f = TraverseNestedDataStructure(keys=parse_path(k))

        # extract_object = lambda objs: traverse_nested_data_structure(data=objs, keys=parse_path(k))
        options[option_key] = f

    return options


def parse_path(path: str) -> List[Union[str, int]]:
    """Parse the path to an object into a list of indexes.

    >>> parse_path("field1.field12[0]") # parsing path to {"field1": {"field12": [<object>, ...], ...}, ...}
    ["field1", "field12", 0]  # data["field1"]["field12"][0] = <object>

    :param path: Path to the object as a string
    """
    pattern = r"\.|\[(\d+)\]"

    result = re.split(pattern, path)
    result = [int(x) if x.isdigit() else x for x in result if x and x != "."]

    return result


def objects_mapping(obj: Any, path: str, targets: List[Tuple[str, str]]) -> Any:
    """Recursive function for "digging" into the mapping object it received and save a "path" to the target.
    Target is defined as one of [torch.Tensor, np.ndarray, PIL.Image]. If got Mapping / Sequence -> continue recursion.

    :param obj:     Recursively returned object
    :param path:    Current path - not achieved a target yet
    :param targets: List of tuples (path.to.object, object_type)
    """
    if isinstance(obj, Mapping):
        printable_map = {}
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            printable_map[k] = objects_mapping(v, new_path, targets)
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        printable_map = []
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            printable_map.append(objects_mapping(v, new_path, targets))
    elif isinstance(obj, str):
        return "string"
    elif isinstance(obj, torch.Tensor):
        printable_map = "Tensor"
        targets.append((path, printable_map))
    elif isinstance(obj, ndarray):
        printable_map = "ndarray"
        targets.append((path, printable_map))
    elif isinstance(obj, Image.Image):
        printable_map = "PIL Image"
        targets.append((path, printable_map))
    else:
        raise RuntimeError(
            f"Unsupported object! Object found has a type of {type(obj)} which is not supported for now.\n"
            f"Supported types: [Mapping, Sequence, String, Tensor, Numpy array, PIL Image]"
        )
    return printable_map


class TraverseNestedDataStructure:
    def __init__(self, keys: List[Union[str, int]]):
        self.keys = keys

    def __call__(self, data: Any) -> Any:
        return traverse_nested_data_structure(data=data, keys=self.keys)


def traverse_nested_data_structure(data: Mapping, keys: List[str]) -> Any:
    """Traverse a nested data structure and returns the value at the specified key path.

    :param data:    Nested data structure like dict, defaultdict or OrderedDict
    :param keys:    List of strings representing the keys in the data structure
    :return:        Value at the specified key path in the data structure
    """
    for key in keys:
        data = data[key]
    return data
