from torch import Tensor


from data_gradients.utils import SegmentationBatchData
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor
from data_gradients.batch_processors.preprocessors.contours import get_contours


class SegmentationBatchPreprocessor(BatchPreprocessor):
    def preprocess(self, images: Tensor, labels: Tensor) -> SegmentationBatchData:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.

        :param images:  Batch of images already formatted into (BS, C, H, W)
        :param labels:  Batch of labels already formatted into (BS, N, W, H)
        :return:        Ready to analyse segmentation batch object.
        """
        contours = [get_contours(onehot_label) for onehot_label in labels]
        return SegmentationBatchData(images=images, labels=labels, contours=contours, split="")
