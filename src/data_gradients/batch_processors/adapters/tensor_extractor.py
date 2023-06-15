from typing import Mapping, Optional, Any, List, Tuple, Sequence, Union
import json
import re

from PIL import Image
import torch
from numpy import ndarray
from torch import Tensor
from data_gradients.utils.utils import ask_user


class TensorExtractor:
    """Extract the tensor of interest (could be image, label, ..) out of a batch raw output (coming from dataloader output).
    This is done by asking the user what field is the relevant one.
    """

    def __init__(self, objs: Any, name: str):
        self.path_to_tensor: Optional[List[str]] = self.prompt_user_for_data_keys(objs=objs, name=name)

    def __call__(self, objs: Any) -> Tensor:
        return self.traverse_nested_data_structure(data=objs, keys=self.path_to_tensor)

    @staticmethod
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

    @staticmethod
    def prompt_user_for_data_keys(objs: Any, name: str) -> List[str]:
        """Extract out of objs all the potential fields of type [torch.Tensor, np.ndarray, PIL.Image], and then
        asks the user to input which of the above keys mapping is the right one in order to retrieve the correct data (either images or labels).

        :param objs:        Dictionary of json-like structure.
        :param name:        The type of your targeted field ('image', 'label', ...). This is only for display purpose.
        :return:            List of keys that if you iterate with the Get Operation (d[k]) through all of them, you will get the data you intended.
                            e.g. ["field1", "field12", 0]  # objs["field1"]["field12"][0] = <object>
        """

        paths = []
        printable_mapping = TensorExtractor.objects_mapping(objs, path="", targets=paths)
        printable_mapping = json.dumps(printable_mapping, indent=4)
        printable_mapping = "This is the structure of your data: \ndata = " + printable_mapping
        main_question = f"Which object maps to your {name} ?"

        options = [f"- {name} = data{k}: {v}" for k, v in paths]
        selected_option = ask_user(main_question=main_question, options=options, optional_description=printable_mapping)

        start_index = selected_option.find("data") + len("data")
        end_index = selected_option.find(":", start_index)
        selected_path = selected_option[start_index:end_index].strip()

        keys = TensorExtractor.parse_path(selected_path)
        return keys

    @staticmethod
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

    @staticmethod
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
                printable_map[k] = TensorExtractor.objects_mapping(v, new_path, targets)
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            if all(isinstance(v, (int, float)) for v in obj):
                printable_map = "List[float|int]"
                targets.append((path, printable_map))
            elif all(isinstance(v, str) for v in obj):
                printable_map = "List[str]"
                targets.append((path, printable_map))
            else:
                printable_map = []
                for i, v in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    printable_map.append(TensorExtractor.objects_mapping(v, new_path, targets))
        elif isinstance(obj, int):
            printable_map = "int"
            targets.append((path, printable_map))
        elif isinstance(obj, float):
            printable_map = "float"
            targets.append((path, printable_map))
        elif isinstance(obj, str):
            printable_map = "String"
            targets.append((path, printable_map))
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

    @staticmethod
    def traverse_nested_data_structure(data: Mapping, keys: List[str]) -> Any:
        """Traverse a nested data structure and returns the value at the specified key path.

        :param data:    Nested data structure like dict, defaultdict or OrderedDict
        :param keys:    List of strings representing the keys in the data structure
        :return:        Value at the specified key path in the data structure
        """
        for key in keys:
            data = data[key]
        return data
