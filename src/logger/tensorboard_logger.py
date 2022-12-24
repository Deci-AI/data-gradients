import torchvision
from torch.utils.tensorboard import SummaryWriter

from src.logger.results_logger import ResultsLogger


class TensorBoardLogger(ResultsLogger):

    def __init__(self, train_iter, samples_to_visualize):
        super().__init__()
        self._train_iter = train_iter
        self.samples_to_visualize = samples_to_visualize
        self.writer = SummaryWriter(log_dir=self.logdir)

    def visualize(self):
        n = 0
        while n < self.samples_to_visualize:
            images, labels = next(self._train_iter)
            n += len(images)
            img_grid = torchvision.utils.make_grid(images)

            # write to tensorboard
            self.writer.add_image(f'{n}_images_labels', img_grid)

    def log(self, title, data):
        title += "/fig"
        self.writer.add_figure(title, data)

    def close(self):
        self.writer.close()
