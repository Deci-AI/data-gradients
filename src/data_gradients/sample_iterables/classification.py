from typing import Iterable, List, Optional
import numpy as np

from data_gradients.utils.data_classes import DetectionSample
from data_gradients.sample_iterables.base import BaseSampleIterable
from data_gradients.utils.data_classes.data_samples import ImageChannelFormat, ClassificationSample
from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter


class ClassificationSampleIterable(BaseSampleIterable):
    def __init__(
        self,
        dataset: ClassificationDatasetAdapter,
        class_names: List[str],
        n_image_channels: int,
        split: str,
        image_format: Optional[ImageChannelFormat] = None,
    ):
        """
        :param class_names: List of all class names in the dataset. The index should represent the class_id.
        """
        super().__init__(dataset=dataset)
        if n_image_channels not in [1, 3]:
            raise ValueError(f"n_image_channels should be either 1 or 3, but got {n_image_channels}")
        self.dataset = dataset
        self.class_names = class_names
        self.n_image_channels = n_image_channels
        self.split = split
        self.image_format = image_format

    def __iter__(self) -> Iterable[DetectionSample]:
        """Group batch images and labels into a single ready-to-analyze batch object, including all relevant preprocessing.
        :return:            Iterable of ready to analyse detection samples.
        """
        for images, labels in self.dataset:
            images = np.uint8(np.transpose(images.cpu().numpy(), (0, 2, 3, 1)))

            if self.image_format is None:
                self.image_format = {1: ImageChannelFormat.GRAYSCALE, 3: ImageChannelFormat.RGB}[self.n_image_channels]

            for image, target in zip(images, labels):
                class_id = int(target)

                sample = ClassificationSample(
                    image=image,
                    class_id=class_id,
                    class_names=self.class_names,
                    split=self.split,
                    image_format=self.image_format,
                    sample_id=None,
                )
                sample.sample_id = str(id(sample))
                yield sample

    def __len__(self) -> int:
        return len(self.dataset)
