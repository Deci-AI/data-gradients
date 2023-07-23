import os
import numpy as np
from typing import Tuple
from torchvision.datasets import CocoDetection


class COCOFormatSegmentationDataset:
    """The Coco Format Segmentation Dataset supports datasets where labels and masks are stored in COCO format.

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
    The annotation files must be structured in JSON format following the COCO data format, including mask data.

    #### Instantiation
    ```python
    from data_gradients.datasets.segmentation import COCOFormatSegmentationDataset
    train_set = COCOFormatSegmentationDataset(
        root_dir="<path/to/dataset_root>",
        images_subdir="images/train",
        annotation_file_path="annotations/train.json"
    )
    val_set = COCOFormatSegmentationDataset(
        root_dir="<path/to/dataset_root>",
        images_subdir="images/validation",
        annotation_file_path="annotations/validation.json"
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
