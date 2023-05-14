from torch import Tensor

from data_gradients.utils import DetectionBatchData
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor


class DetectionBatchPreprocessor(BatchPreprocessor):
    def preprocess(self, images: Tensor, labels: Tensor) -> DetectionBatchData:
        return DetectionBatchData(images=images, labels=labels, split="")
