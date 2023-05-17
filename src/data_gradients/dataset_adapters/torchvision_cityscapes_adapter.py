from typing import List, Tuple

import numpy as np
import torchvision

from data_gradients.dataset_adapters import SegmentationDatasetAdapter, SegmentationSample


class TorchvisionCityscapesSegmentationAdapter(SegmentationDatasetAdapter):
    """
    SegmentationDatasetAdapter implementation of the Cityscapes dataset class from torchvision.
    """

    def __init__(self, dataset: torchvision.datasets.Cityscapes):
        if not isinstance(dataset, torchvision.datasets.Cityscapes):
            raise ValueError("dataset must be an instance of torchvision.datasets.Cityscapes")

        if "semantic" not in dataset.target_type:
            raise ValueError("dataset must be an instance of torchvision.datasets.Cityscapes with target_type='semantic'")

        self.dataset = dataset
        self.class_names = [c.name for c in dataset.classes]

    def get_num_classes(self) -> int:
        return len(self.dataset.classes)

    def get_class_names(self) -> List[str]:
        return [c.name for c in self.dataset.classes]

    def __len__(self) -> int:
        return len(self.dataset)

    def get_ignored_classes(self):
        return []

    def __getitem__(self, index):
        sample = self.dataset[index]
        image, mask = sample[:2]

        image = np.array(image)
        mask = np.array(mask)

        return SegmentationSample(sample_id=self.dataset.images[index], image=image, mask=mask)

    def get_iterator(self):
        num_samples = len(self.dataset)
        for i in range(num_samples):
            sample = self.dataset[i]

            image, mask = sample[:2]

            image = np.array(image)
            mask = np.array(mask)

            yield SegmentationSample(sample_id=self.dataset.images[i], image=image, mask=mask)
