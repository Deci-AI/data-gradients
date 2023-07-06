import os
import numpy as np
from typing import Tuple

from torchvision.datasets import CocoDetection


class CocoFormatDetectionDataset:
    """The Coco Format Detection Dataset supports datasets where labels and annotations are stored in COCO format.

    #### Expected folder structure
    The dataset folder structure should include at least one sub-directory for images and one JSON file for annotations.

    Example:
    ```
        dataset_root/
            ├── images/
            │   ├── train/
            │   │   ├── 1.jpg
            │   │   ├── 2.jpg
            │   │   └── ...
            │   ├── test/
            │   │   ├── ...
            │   └── validation/
            │       ├── ...
            └── annotations/
                ├── train.json
                ├── test.json
                └── validation.json
    ```
    #### Expected Annotation File Structure
    The annotation files must be structured in JSON format following the COCO data format.

    #### Instantiation
    ```
    dataset_root/
        ├── images/
        │   ├── train/
        │   │   ├── 1.jpg
        │   │   ├── 2.jpg
        │   │   └── ...
        │   ├── val/
        │   │   ├── ...
        │   └── test/
        │       ├── ...
        └── annotations/
            ├── train.json
            ├── test.json
            └── validation.json
    ```

    ```python
    from data_gradients.datasets.detection import CocoFormatDetectionDataset

    train_set = CocoFormatDetectionDataset(
        root_dir="<path/to/dataset_root>", images_subdir="images/train", annotation_file_path="annotations/train.json"
    )
    val_set = CocoFormatDetectionDataset(
        root_dir="<path/to/dataset_root>", images_subdir="images/validation", annotation_file_path="annotations/validation.json"
    )
    ```
    """

    def __init__(self, root_dir: str, images_subdir: str, annotation_file_path: str):
        """
        :param root_dir:                Where the data is stored.
        :param images_subdir:           Local path to directory that includes all the images. Path relative to `root_dir`.
        :param annotation_file_path:    Local path to annotation file. Path relative to `root_dir`.
        """

        self.base_dataset = CocoDetection(
            root=os.path.join(root_dir, images_subdir),
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
