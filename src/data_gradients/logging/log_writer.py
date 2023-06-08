import os
import json
from typing import Union, Dict, Any, List


JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


class JsonWriter:
    """Writer responsible for writing data to a json file.
    :param filename: Name of the file to write to.
    """

    def __init__(self, filename: str):
        if not filename.endswith(".json"):
            raise ValueError("`filename` must end with `.json`")
        self.filename = filename

    def write(self, data: JSONValue, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self.filename)
        with open(output_path, "a") as f:
            json.dump(data, f, indent=4)
