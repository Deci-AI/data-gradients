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
        try:
            all_images, all_labels = next(self._train_iter)
            while len(all_images) < self.samples_to_visualize:
                images, labels = next(self._train_iter)
                all_images = torch.concat([all_images, images], dim=0)
                all_labels = torch.concat([all_labels, labels], dim=0)

            if len(all_images) > self.samples_to_visualize:
                all_images = all_images[:self.samples_to_visualize, ...]
                all_labels = all_labels[:self.samples_to_visualize, ...]

            all_labels *= (1. / max(all_labels.unique()))
            all_labels = all_labels.repeat(1, 3, 1, 1)

            if all_images.shape != all_labels.shape:
                resize = torchvision.transforms.Resize(all_labels.shape[-2:])
                all_images = resize(all_images)

            img_grid = torchvision.utils.make_grid(torch.cat([all_images, all_labels]), nrow=len(all_images))
            title = f'Data Visualization/{len(all_images)} Images & Labels'
            self.writer.add_image(title, img_grid)

        except RuntimeError as e:
            print(f'\nCould not visualize images on tensorboard\n'
                  f'Images shape: {all_images.shape} , Labels shape: {all_labels.shape}\n'
                  f'{e}')

    def log(self, title, data):
        # title += "/fig"
        self.writer.add_figure(title, data)

    def close(self):
        self.writer.close()
