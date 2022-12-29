import os
import json

from src.logger.results_logger import ResultsLogger


class JsonLogger(ResultsLogger):

    def __init__(self):
        super().__init__()
        self._output_file = 'raw_data.json'

    def log(self, title: str, data):
        with open(os.path.join(self.logdir, self._output_file), 'w') as output:
            try:
                json.dump(data, output, indent=2)
            except Exception as e:
                print(f'Could not serialize {title}!'
                      f'\nObject type is {type(data)}'
                      f'\nException is\n{e}')

    def close(self):
        pass

