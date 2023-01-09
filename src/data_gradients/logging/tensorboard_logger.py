import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from data_gradients.logging.results_logger import ResultsLogger


class TensorBoardLogger(ResultsLogger):

    def __init__(self, train_iter, samples_to_visualize):
        super().__init__()
        self._train_iter = train_iter
        self.samples_to_visualize = samples_to_visualize
        self.writer = SummaryWriter(log_dir=self.logdir)

    def visualize(self):
        if self.samples_to_visualize == 0:
            return
        n = 0
        while n < self.samples_to_visualize:
            # TODO: Still WIP
            images, labels = next(self._train_iter)
            if len(images) > self.samples_to_visualize:
                images = images[:self.samples_to_visualize]
                labels = labels[:self.samples_to_visualize]

            labels *= (1. / max(labels.unique()))
            labels = labels.repeat(1, 3, 1, 1)

            img_grid = torchvision.utils.make_grid(torch.cat([images, labels]), nrow=len(images))

            n += len(images)

            # write to tensorboard
            self.writer.add_image(f'{n}_images_labels', img_grid)

    def log(self, title, data):
        title += "/fig"
        self.writer.add_figure(title, data)

    def close(self):
        self.writer.close()
