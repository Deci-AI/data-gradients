import os
import json
from typing import Dict

from src.logging.results_logger import ResultsLogger


class JsonLogger(ResultsLogger):

    def __init__(self):
        super().__init__()
        self._output_file = 'raw_data.json'
        self._logging_data = {}

    def log(self, title: str, data):
        self._logging_data.update({title: data})

    def write_to_json(self):
        with open(os.path.join(self.logdir, self._output_file), 'a') as output:
            try:
                json.dump(self._logging_data, output, indent=4)
            except Exception as e:
                print(e)

    def close(self):
        pass

