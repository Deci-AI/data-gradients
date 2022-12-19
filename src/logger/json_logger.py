import os
import json

from src.logger.results_logger import ResultsLogger


class JsonLogger(ResultsLogger):

    def __init__(self):
        super().__init__()
        self._output_file = 'raw_data.json'
        self._json_object = dict()

    def log(self, title: str, data):
        self._json_object.update({"Train-" + title: data[0],
                                  "Val-" + title: data[1]})
        with open(os.path.join(self.logdir, self._output_file), 'w') as output:
            json.dump(self._json_object, output, indent=2)

    def close(self):
        pass

