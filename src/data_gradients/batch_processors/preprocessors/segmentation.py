from typing import Optional, Dict, Iterable
import time
import numpy as np
from torch import Tensor


from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor
from data_gradients.batch_processors.preprocessors.contours import get_contours
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat


class SegmentationBatchPreprocessor(BatchPreprocessor):
    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        :param class_names: Mapping of ids to class names. Ids not mapped will be ignored.
        """
        self.class_names = class_names

    def preprocess(self, images: Tensor, labels: Tensor, split: str) -> Iterable[SegmentationSample]:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.

        :param images:      Batch of images already formatted into (BS, C, H, W)
        :param labels:      Batch of labels already formatted into (BS, N, H, W)
        :param split:       Name of the split (train, val, test)
        :return:            Ready to analyse segmentation batch object.
        """
        images = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))
        labels = labels.cpu().numpy()

        for image, mask in zip(images, labels):
            contours = get_contours(mask)

            # TODO: image_format is hard-coded here, but it should be refactored afterwards
            yield SegmentationSample(
                image=image,
                mask=mask,
                contours=contours,
                class_names=self.class_names,
                split=split,
                image_format=ImageChannelFormat.RGB,
                sample_id=str(time.time()),
            )
