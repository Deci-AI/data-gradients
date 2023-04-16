import abc
from torch.utils.tensorboard import SummaryWriter

from data_gradients.logging.results_logger import ResultsLogger
from data_gradients.utils import BatchData


class TensorBoardLogger(ResultsLogger):
    def __init__(self, samples_to_visualize):
        super().__init__()
        self.remaining_samples_to_visualize = samples_to_visualize
        self.writer = SummaryWriter(log_dir=self.logdir)

    def visualize(self, batch_data: BatchData):
        if self.remaining_samples_to_visualize <= 0:
            return
        try:
            n_visualized = self._visualize(batch_data)
            self.remaining_samples_to_visualize = (
                self.remaining_samples_to_visualize - n_visualized
            )
        except RuntimeError as e:
            print(f"\nCould not visualize images on tensorboard\n: {e}")

    @abc.abstractmethod
    def _visualize(self, samples: BatchData):
        raise NotImplementedError

    def log(self, title, data):
        # title += "/fig"
        self.writer.add_figure(title, data)

    def close(self):
        self.writer.close()
