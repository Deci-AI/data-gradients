
from torch.utils.tensorboard import SummaryWriter

from src.logger.results_logger import ResultsLogger


class TensorBoardLogger(ResultsLogger):

    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter(log_dir=self.logdir)

    def log_graph(self, title, plot):
        self.writer.add_figure(title, plot)

    def log_text(self, title, text):
        self.writer.add_image(title, text)

    def close(self):
        self.writer.close()
