from typing import List, Iterable
import time
import numpy as np

from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.sample_iterable.base import BaseSampleIterable
from data_gradients.sample_iterable.contours import get_contours
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat
from data_gradients.dataset_adapter.segmentation_adapter import SegmentationDatasetAdapter


class SegmentationSampleIterable(BaseSampleIterable):
    def __init__(
        self,
        dataset: SegmentationDatasetAdapter,
        class_names: List[str],
        split: str,
        image_format: ImageChannelFormat = ImageChannelFormat.RGB,
    ):
        """
        :param class_names: List of all class names in the dataset. The index should represent the class_id.
        """
        super().__init__(dataset=dataset)
        self.dataset = dataset
        self.class_names = class_names
        self.split = split
        self.image_format = image_format

    def __iter__(self) -> Iterable[SegmentationSample]:
        for images, labels in self.dataset:
            images = np.uint8(np.transpose(images.cpu().numpy(), (0, 2, 3, 1)))
            labels = np.uint8(labels.cpu().numpy())

            for image, mask in zip(images, labels):
                contours = get_contours(mask)

                yield SegmentationSample(
                    image=image,
                    mask=mask,
                    contours=contours,
                    class_names=self.class_names,
                    split=self.split,
                    image_format=self.image_format,
                    sample_id=str(time.time()),
                )

    def __len__(self) -> int:
        return len(self.dataset)
