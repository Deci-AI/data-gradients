import os
import json

from data_gradients.logging.results_logger import ResultsLogger


class JsonLogger(ResultsLogger):
    def __init__(self, output_file_name: str):
        super().__init__()
        self._logging_data = {}
        self.output_file_name = output_file_name + ".json"

    def log(self, title: str, data):
        self._logging_data.update({title: data})

    def write_to_json(self):
        with open(os.path.join(self.logdir, self.output_file_name), "a") as output:
            try:
                json.dump(self._logging_data, output, indent=4)
            except Exception as e:
                print(e)

    def close(self):
        pass
