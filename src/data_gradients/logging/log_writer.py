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
    """Class for logging data in TensorBoard and JSON formats."""

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the LogWriter object.

        :param log_dir: Optional, Directory to save the log files.
                        If None, the logs will be saved in a 'logs' directory in the current working directory.
        """
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
            logger.info(f"`log_dir` was not set, so the logs will be saved in {log_dir}")

        session_dir_name = "log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir, session_dir_name)

        self.log_dir = log_dir
        self._tb_logger = TensorBoardLogger(log_dir=log_dir)
        self._json_logger = JsonLogger(log_dir=log_dir, output_file_name="raw_data")

    def log_json(self, title: str, data: JSONValue) -> None:
        """Log data in JSON format.

        :param title:   Title of the data to be logged.
        :param data:    Data to be logged in JSON format.
        """
        self._json_logger.log(title=title, data=data)

    def log_figure(self, title: str, figure: Figure) -> None:
        """Log a figure to TensorBoard.

        :param title:   Title of the data to be logged.
        :param figure:  Figure to be logged to the TensorBoard.
        """
        self._tb_logger.log(title=title, data=figure)

    def log_image(self, title: str, image: torch.Tensor) -> None:
        """
        Log an image in TensorBoard format.

        :param title:   Title of the data to be logged.
        :param image:   Image to be logged to TensorBoard.
        """
        self._tb_logger.log_image(title=title, image=image)

    def save_as_json(self) -> None:
        """Save the logged data in JSON format."""
        self._json_logger.save_as_json()

    def close(self) -> None:
        """Close the TensorBoard and JSON loggers."""
        self._json_logger.close()
        self._tb_logger.close()
