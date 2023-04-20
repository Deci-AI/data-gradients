import torch
import torchvision

from data_gradients.logging.tensorboard_logger import TensorBoardLogger
from data_gradients.utils import SegBatchData


class SegmentationTensorBoardLogger(TensorBoardLogger):
    def _visualize(self, batch_data: SegBatchData):
        images = list(batch_data.images)
        labels = list(batch_data.labels)
        num_samples_to_visualize = min(len(images), self.remaining_samples_to_visualize)
        for i in range(num_samples_to_visualize):
            image = images[i]
            label = labels[i]
            if label.shape[2:] != image.shape[2:]:  # Just in case label is different from image
                resize = torchvision.transforms.Resize(label.shape[-2:])
                image = resize(image)

            idxs = torch.arange(len(label)).view(len(label), 1, 1) + 1  # +1 to avoid 'killing' channel zero
            label = idxs * label
            label = torch.sum(label, dim=0)
            label = torch.ceil((label * 255) / torch.max(label) + 1)  # normalize such that max(label) goes to 255
            label = label.repeat(3, 1, 1)  # to rgb (with same values -> grayscale)
            image_and_label = torch.cat((image.float(), label), dim=-1)
            title = f"Data Visualization/{self.remaining_samples_to_visualize - i}"
            self.writer.add_image(title, image_and_label)

        return num_samples_to_visualize
