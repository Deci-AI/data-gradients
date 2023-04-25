from torch import Tensor

from data_gradients.utils import DetectionBatchData
from data_gradients.batch_processors.preprocessors.base import Preprocessor


class DetectionBatchPreprocessor(Preprocessor):
    def __call__(self, images: Tensor, labels: Tensor) -> DetectionBatchData:
        return DetectionBatchData(images=images, labels=labels, split="")
