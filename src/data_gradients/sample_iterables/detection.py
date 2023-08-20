from typing import Iterable, List
import numpy as np
import time

from data_gradients.utils.data_classes import DetectionSample
from data_gradients.sample_iterables.base import BaseSampleIterable
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter


class DetectionSampleIterable(BaseSampleIterable):
    def __init__(
        self,
        dataset: DetectionDatasetAdapter,
        class_names: List[str],
        split: str,
        image_format: ImageChannelFormat = ImageChannelFormat.GRAYSCALE,
    ):
        """
        :param dataset: Dataset Adapter to iterate over.
        :param class_names: List of all class names in the dataset. The index should represent the class_id.
        :param split: Dataset split. ("train", "val")
        :param image_format: Format of the images. ("RGB", "BGR", "GRAYSCALE", ...)
        """
        super().__init__(dataset=dataset)
        self.dataset = dataset
        self.class_names = class_names
        self.split = split
        self.image_format = image_format

    def __iter__(self) -> Iterable[DetectionSample]:
        for images, labels in self.dataset:
            images = np.uint8(np.transpose(images.cpu().numpy(), (0, 2, 3, 1)))

            for image, target in zip(images, labels):
                target = target.cpu().numpy().astype(int)
                class_ids, bboxes_xyxy = target[:, 0], target[:, 1:]

                yield DetectionSample(
                    image=image,
                    class_ids=class_ids,
                    bboxes_xyxy=bboxes_xyxy,
                    class_names=self.class_names,
                    split=self.split,
                    image_format=self.image_format,
                    sample_id=str(time.time()),
                )

    def __len__(self) -> int:
        return len(self.dataset)
