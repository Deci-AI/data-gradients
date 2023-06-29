import os
import numpy as np
from typing import Tuple

from torchvision.datasets import CocoDetection


class CocoFormatDetectionDataset:
    def __init__(self, root_dir: str, images_dir: str, annotation_file_path: str):
        """
        :param root_dir:                Where the data is stored.
        :param images_dir:              Local path to directory that includes all the images. Path relative to `root_dir`.
        :param annotation_file_path:    Local path to annotation file. Path relative to `root_dir`.
        """

        self.base_dataset = CocoDetection(
            root=os.path.join(root_dir, images_dir),
            annFile=os.path.join(root_dir, annotation_file_path),
        )

    def load_image(self, index: int) -> np.ndarray:
        image = self.base_dataset[index][0]
        return np.array(image)

    def load_labels(self, index: int) -> np.ndarray:
        annotations = self.base_dataset[index][1]

        labels = []
        for annotation in annotations:
            class_id = annotation["category_id"]
            x, y, w, h = annotation["bbox"]
            labels.append((int(class_id), float(x), float(y), float(w), float(h)))
        return np.array(labels)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(index)
        labels = self.load_labels(index)
        return image, labels

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(len(self)):
            yield self[i]
