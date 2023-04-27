import os
import json
from typing import Union, Dict, Any, List

from data_gradients.logging.loggers.results_logger import ResultsLogger

JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


class JsonLogger(ResultsLogger):
    def __init__(self, log_dir, output_file_name: str):
        super().__init__(log_dir=log_dir)
        self._logging_data = {}
        self.output_path = os.path.join(log_dir, output_file_name + ".json")

    def log(self, title: str, data: JSONValue) -> None:
        self._logging_data.update({title: data})

    def save_as_json(self) -> None:
        with open(self.output_path, "a") as output:
            try:
                json.dump(self._logging_data, output, indent=4)
            except Exception as e:
                print(e)

    def close(self) -> None:
        pass
