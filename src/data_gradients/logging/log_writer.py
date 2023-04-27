import os
import datetime
from typing import Optional
import logging

from matplotlib.figure import Figure
import torch

from data_gradients.logging.loggers.json_logger import JsonLogger, JSONValue
from data_gradients.logging.loggers.tensorboard_logger import TensorBoardLogger


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class LogWriter:
    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
            logger.info(f"`log_dir` was not set, so the logs will be saved in {log_dir}")

        session_dir_name = "log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir, session_dir_name)

        self.log_dir = log_dir
        self._tb_logger = TensorBoardLogger(log_dir=log_dir)
        self._json_logger = JsonLogger(log_dir=log_dir, output_file_name="raw_data")

    def log_json(self, title: str, data: JSONValue) -> None:
        self._json_logger.log(title=title, data=data)

    def log_tb(self, title: str, data: Figure) -> None:
        self._tb_logger.log(title=title, data=data)

    def log(self, title: str, figure: Optional[Figure] = None, json_data: Optional[JSONValue] = None):
        if figure is not None:
            self._tb_logger.log(title=title, data=figure)
        if json_data is not None:
            self._json_logger.log(title=title, data=json_data)

    def log_image(self, title: str, image: torch.Tensor) -> None:
        self._tb_logger.log_image(title=title, image=image)

    def save_as_json(self):
        self._json_logger.save_as_json()

    def close(self):
        self._json_logger.close()
        self._tb_logger.close()
        print(
            f'{"*" * 100}'
            f"\nWe have finished evaluating your dataset!"
            f"\nThe results can be seen in {self.log_dir}"
            f"\n\nShow tensorboard by writing in terminal:"
            f"\n\ttensorboard --logdir={self.log_dir} --bind_all"
            f"\n"
        )
