from typing import Iterable, List
from torch import Tensor
import numpy as np
import time

from data_gradients.utils.data_classes import DetectionSample
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat


class DetectionBatchPreprocessor(BatchPreprocessor):
    def __init__(self, class_names: List[str]):
        """
        :param class_names: List of all class names in the dataset. The index should represent the class_id.
        """
        self.class_names = class_names

    def preprocess(self, images: Tensor, labels: Tensor, split: str) -> Iterable[DetectionSample]:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.

        :param images:      Batch of images already formatted into (BS, C, H, W)
        :param labels:      List of bounding boxes, each of shape (N_i, 5 [label_xyxy]) with N_i being the number of bounding boxes with class_id in class_ids
        :param split:       Name of the split (train, val, test)
        :return:            Iterable of ready to analyse detection samples.
        """
        images = np.uint8(np.transpose(images.cpu().numpy(), (0, 2, 3, 1)))

        for image, target in zip(images, labels):
            target = target.cpu().numpy().astype(int)
            class_ids, bboxes_xyxy = target[:, 0], target[:, 1:]

            # TODO: image_format is hard-coded here, but it should be refactored afterwards
            yield DetectionSample(
                image=image,
                class_ids=class_ids,
                bboxes_xyxy=bboxes_xyxy,
                class_names=self.class_names,
                split=split,
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
        is_padded_index = np.where((target == padding_value).all(axis=1))[0]
        if len(is_padded_index) > 0:
            target = target[is_padded_index]
        return target
