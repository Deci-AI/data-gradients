import torch
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

from data_gradients.logging.loggers.results_logger import ResultsLogger


class TensorBoardLogger(ResultsLogger):
    def __init__(self, log_dir: str):
        super().__init__(log_dir=log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_image(self, title: str, image: torch.Tensor) -> None:
        """Log an image to TensorBoard.

        :param title:   Title of the data to be logged.
        :param image:   Image to be logged to the TensorBoard.
        """
        self.writer.add_image(tag=title, img_tensor=image)

    def log(self, title: str, data: Figure) -> None:
        """Log a figure to TensorBoard.

        :param title:   Title of the data to be logged.
        :param data:    Figure to be logged to the TensorBoard.
        """
        self.writer.add_figure(title, data)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
