from typing import Mapping, Dict, Any, List, Tuple, Sequence, Union
import re

from PIL import Image
import torch
from numpy import ndarray


def get_tensor_extractor_options(objs: Any) -> Dict[str, str]:
    """Extract out of objs all the potential fields of type [torch.Tensor, np.ndarray, PIL.Image], and then
    asks the user to input which of the above keys mapping is the right one in order to retrieve the correct data (either images or labels).

    :param objs: Dictionary following the pattern: {"path.to.object: object_type": "path.to.object"}
    """
    objects_mapping: List[Tuple[str, str]] = []  # Placeholder for list of (path.to.object, object_type)
    extract_object_mapping(objs, current_path="", objects_mapping=objects_mapping)
    options = {f"{path_to_object}: {object_type}": path_to_object for path_to_object, object_type in objects_mapping}
    return options


def extract_object_mapping(current_object: Any, current_path: str, objects_mapping: List[Tuple[str, str]]) -> Any:
    """Recursive function for "digging" into the mapping object it received and save a "path" to the target.
    Target is defined as one of [torch.Tensor, np.ndarray, PIL.Image]. If got Mapping / Sequence -> continue recursion.

    :param current_object:  Recursively returned object
    :param current_path:    Current path - not achieved a target yet
    :param objects_mapping: List of tuples (path.to.object, object_type)
    """
    if isinstance(current_object, Mapping):
        printable_map = {}
        for k, v in current_object.items():
            new_path = f"{current_path}.{k}" if current_path else k
            printable_map[k] = extract_object_mapping(v, new_path, objects_mapping)
    elif isinstance(current_object, Sequence) and not isinstance(current_object, str):
        if all(isinstance(v, (int, float)) for v in current_object):
            printable_map = "List[float|int]"
            objects_mapping.append((current_path, printable_map))
        elif all(isinstance(v, str) for v in current_object):
            printable_map = "List[str]"
            objects_mapping.append((current_path, printable_map))
        else:
            printable_map = []
            for i, v in enumerate(current_object):
                new_path = f"{current_path}[{i}]"
                printable_map.append(extract_object_mapping(v, new_path, objects_mapping))
    elif isinstance(current_object, int):
        printable_map = "int"
        objects_mapping.append((current_path, printable_map))
    elif isinstance(current_object, float):
        printable_map = "float"
        objects_mapping.append((current_path, printable_map))
    elif isinstance(current_object, str):
        printable_map = "String"
        objects_mapping.append((current_path, printable_map))
    elif isinstance(current_object, torch.Tensor):
        printable_map = "Tensor"
        objects_mapping.append((current_path, printable_map))
    elif isinstance(current_object, ndarray):
        printable_map = "ndarray"
        objects_mapping.append((current_path, printable_map))
    elif isinstance(current_object, Image.Image):
        printable_map = "PIL Image"
        objects_mapping.append((current_path, printable_map))
    else:
        raise RuntimeError(
            f"Unsupported object! Object found has a type of {type(current_object)} which is not supported for now.\n"
            f"Supported types: [Mapping, Sequence, String, Tensor, Numpy array, PIL Image]"
        )
    return printable_map


class DataLookupError(Exception):
    def __init__(self, e):
        super().__init__(e)


class NestedDataLookup:
    """Callable that allows to traverse a data structure according to an input path."""

    def __init__(self, object_path: str):
        self.keys_to_reach_object = extract_keys_from_path(object_path=object_path)

    def __call__(self, data: Any) -> Any:
        try:
            return traverse_nested_data_structure(data=data, keys=self.keys_to_reach_object)
        except Exception as e:
            raise DataLookupError(e)


def extract_keys_from_path(object_path: str) -> List[Union[str, int]]:
    """Parse the path to an object into a list of indexes.


    >> extract_keys_from_path("field1.field12[0]") # Which originally represents {"field1": {"field12": [<object>, ...], ...}, ...}
    ["field1", "field12", 0]  # Can be used like this: data["field1"]["field12"][0] = <object>

    :param object_path: Path to the object as a string
    """
    pattern = r"\.|\[(\d+)\]"

    result = re.split(pattern, object_path)
    result = [int(x) if x.isdigit() else x for x in result if x and x != "."]

    return result


def traverse_nested_data_structure(data: Union[List, Mapping], keys: List[str]) -> Any:
    """Traverse a nested data structure and returns the value at the specified key path.

    :param data:    Nested data structure like dict, defaultdict or OrderedDict
    :param keys:    List of strings representing the keys in the data structure
    :return:        Value at the specified key path in the data structure
    """
    for key in keys:
        data = data[key]
    return data
