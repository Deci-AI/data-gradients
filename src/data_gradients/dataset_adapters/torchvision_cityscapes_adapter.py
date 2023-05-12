from typing import List, Tuple

import numpy as np
import torchvision

from data_gradients.dataset_adapters import SegmentationDatasetAdapter


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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        sample = self.dataset[index]
        image, mask = sample
        return np.array(image), np.array(mask)

    def get_num_classes(self) -> int:
        return len(self.dataset.classes)

    def get_class_names(self) -> List[str]:
        return self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)
