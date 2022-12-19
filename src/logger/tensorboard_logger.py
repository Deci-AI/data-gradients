
from torch.utils.tensorboard import SummaryWriter

from src.logger.results_logger import ResultsLogger


class TensorBoardLogger(ResultsLogger):

    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter(log_dir=self.logdir)

    def log(self, title, data):
        title += "/fig"
        self.writer.add_figure(title, data)

    def close(self):
        self.writer.close()
