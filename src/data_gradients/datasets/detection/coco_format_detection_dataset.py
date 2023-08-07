import os
import numpy as np
from typing import Tuple

from torchvision.datasets import CocoDetection


class COCOFormatDetectionDataset:
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
    from data_gradients.datasets.detection import COCOFormatDetectionDataset

    train_set = COCOFormatDetectionDataset(
        root_dir="<path/to/dataset_root>", images_subdir="images/train", annotation_file_path="annotations/train.json"
    )
    val_set = COCOFormatDetectionDataset(
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

        self.class_ids = self.base_dataset.coco.getCatIds()

        categories = self.base_dataset.coco.loadCats(self.class_ids)
        self.class_names = [category["name"] for category in categories]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image, annotations = self.base_dataset[index]
        image = np.array(image)

        labels = []
        for annotation in annotations:
            class_id = annotation["category_id"]
            x, y, w, h = annotation["bbox"]
            labels.append((int(class_id), float(x), float(y), float(w), float(h)))

        labels = np.array(labels, dtype=np.float32).reshape(-1, 5)

        return image, labels
