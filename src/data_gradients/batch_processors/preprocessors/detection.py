from typing import Iterable
from torch import Tensor
import numpy as np
import time

from data_gradients.utils.data_classes import DetectionSample
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat


class DetectionBatchPreprocessor(BatchPreprocessor):
    def preprocess(self, images: Tensor, labels: Tensor) -> Iterable[DetectionSample]:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.

        :param images:  Batch of images already formatted into (BS, C, H, W)
        :param labels:  Batch of labels already formatted into (BS, N, 5), in format (class_id, x1, y1, x2, y2)
        :return:        Iterable of ready to analyse detection samples.
        """
        images = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))
        labels = labels.cpu().numpy()

        for image, target in zip(images, labels):
            target = self.filter_padding(target, padding_value=0).astype(np.int)
            class_ids, bboxes_xyxy = target[:, 0], target[:, 1:]

            # TODO: image_format is hard-coded here, but it should be refactored afterwards
            yield DetectionSample(
                image=image,
                class_ids=class_ids,
                bboxes_xyxy=bboxes_xyxy,
                split=None,
                image_format=ImageChannelFormat.RGB,
                sample_id=str(time.time()),
            )

    @staticmethod
    def filter_padding(target: np.ndarray, padding_value: int) -> np.ndarray:
        """Drop rows that were padded with padding_value.

        :target:        Target bboxes of a given image, in shape (K, ?) with K either number of bboxes for this image, or padding size,
        :padding_value: Value used for padding (if any)
        :return:        Filtered bboxes of a given image, in shape (N, ?) with N number of bboxes for this image.
        """
        first_zero_row_index = np.where((target == padding_value).all(axis=1))[0][0]
        return target[:first_zero_row_index]
