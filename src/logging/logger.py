from typing import Dict

from src.logging.json_logger import JsonLogger
from src.logging.tensorboard_logger import TensorBoardLogger


class Logger:
    def __init__(self, samples_to_visualize, train_data):
        self._tb_logger = TensorBoardLogger(iter(train_data), samples_to_visualize)
        self._json_logger = JsonLogger()

    def visualize(self):
        self._tb_logger.visualize()

    def log(self, title_name: str, tb_data=None, json_data=None):
        if tb_data is not None:
            self._tb_logger.log(title_name, tb_data)
        if json_data is not None:
            self._json_logger.log(title_name, json_data)

    def log_meta_data(self, route: Dict):
        self._json_logger.log('Get data out of dictionary', route)

    def to_json(self):
        self._json_logger.write_to_json()

    def close(self):
        self._json_logger.close()
        self._tb_logger.close()

    def results_dir(self):
        return self._json_logger.logdir
