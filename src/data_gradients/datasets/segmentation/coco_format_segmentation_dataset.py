import os
import numpy as np
from typing import Tuple
from torchvision.datasets import CocoDetection


class CocoFormatSegmentationDataset:
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

        self.class_ids = self.base_dataset.coco.getCatIds()

        categories = self.base_dataset.coco.loadCats(self.class_ids)
        self.class_names = [category["name"] for category in categories]

    def load_image(self, index: int) -> np.ndarray:
        image = self.base_dataset[index][0]
        return np.array(image)

    def load_masks(self, index: int) -> np.ndarray:
        annotations = self.base_dataset[index][1]

        # Get image size
        image_size = self.base_dataset[index][0].size[::-1]

        # Initialize empty mask
        masks = np.zeros(image_size, dtype=np.uint8)

        for annotation in annotations:
            class_id = annotation["category_id"]
            mapped_id = self.class_ids.index(class_id)  # map original class ID to a continuous sequence
            mask = self.base_dataset.coco.annToMask(annotation)  # Getting mask for each annotation
            masks[mask > 0] = mapped_id  # Set the corresponding class ID in the mask

        return masks

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(index)
        masks = self.load_masks(index)
        return image, masks

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(len(self)):
            yield self[i]
