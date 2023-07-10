from typing import Tuple

import numpy as np
from torchvision.transforms import transforms

from data_gradients.datasets.base_dataset import BaseImageLabelDirectoryDataset
from data_gradients.datasets.utils import load_image, ImageChannelFormat

NORMALIZATION_MEANS = [0.485, 0.456, 0.406]
NORMALIZATION_STDS = [0.229, 0.224, 0.225]

BDDD_CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "<?>",
]


class BDDDataset(BaseImageLabelDirectoryDataset):
    """
    PyTorch Dataset implementation of the BDD100K dataset.
    The BDD100K data and annotations can be obtained at https://bdd-data.berkeley.edu/.
    """

    CLASS_NAMES = BDDD_CLASS_NAMES

    def __init__(
        self,
        data_folder,
        split: str,
        transform=transforms.Compose([]).transforms,
        verbose: bool = False,
    ):
        """
        :param data_folder: Folder where data files are stored
        :param split: 'train' or 'test'
        """
        self.transforms = transform
        self.target_transforms = transform

        super().__init__(
            root_dir=data_folder,
            images_subdir=split,
            labels_subdir=split,
            verbose=verbose,
            image_extensions=("jpg",),
            label_extensions=("png",),
        )

    def load_labels(self, path: str) -> np.ndarray:
        return load_image(path, ImageChannelFormat.UNCHANGED)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image, label = super().__getitem__(index)
        if self.transforms:
            image = self.transforms(image)
            label = self.transforms(label)
        return image, label