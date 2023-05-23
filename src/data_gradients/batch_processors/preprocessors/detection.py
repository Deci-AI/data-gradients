from torch import Tensor
import numpy as np

from data_gradients.utils.data_classes import DetectionSample
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat


class DetectionBatchPreprocessor(BatchPreprocessor):
    def preprocess(self, images: Tensor, labels: Tensor) -> DetectionSample:
        images = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))
        labels = labels.cpu().numpy()

        for image, target in zip(images, labels):
            # TODO: image_format is hard-coded here, but it should be refactored afterwards
            yield DetectionSample(image=image, target=target, split=None, image_format=ImageChannelFormat.RGB, sample_id=None)
