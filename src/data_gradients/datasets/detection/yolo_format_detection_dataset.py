import os
import numpy as np
import logging
from typing import Tuple, Sequence

from data_gradients.datasets.FolderProcessor import ImageLabelFilesIterator, DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image, ImageChannelFormat


logger = logging.getLogger(__name__)


class YoloFormatDetectionDataset:
    """The Yolo format Detection Dataset supports any dataset stored in the YOLO format.

    #### Expected folder structure
    Any structure including at least one sub-directory for images and one for labels. They can be the same.

    Example 1: Separate directories for images and labels
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
            └── labels/
                ├── train/
                │   ├── 1.txt
                │   ├── 2.txt
                │   └── ...
                ├── test/
                │   ├── ...
                └── validation/
                    ├── ...
    ```

    Example 2: Same directory for images and labels
    ```
        dataset_root/
            ├── train/
            │   ├── 1.jpg
            │   ├── 1.txt
            │   ├── 2.jpg
            │   ├── 2.txt
            │   └── ...
            └── validation/
                ├── ...
    ```

    #### Expected label files structure
    The label files must be structured such that each row represents a bounding box label.
    Each bounding box is represented by 5 elements: `class_id, cx, cy, w, h`.

    #### Instantiation
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
        └── labels/
            ├── train/
            │   ├── 1.txt
            │   ├── 2.txt
            │   └── ...
            ├── test/
            │   ├── ...
            └── validation/
                ├── ...
    ```

    ```python
    from data_gradients.datasets.detection import YoloFormatDetectionDataset

    train_loader = YoloFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
    val_loader = YoloFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
    ```

    This class does NOT support dataset formats such as Pascal VOC or COCO.
    """

    def __init__(
        self,
        root_dir: str,
        images_dir: str,
        labels_dir: str,
        ignore_invalid_labels: bool = True,
        verbose: bool = False,
        image_extensions: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        label_extensions: Sequence[str] = ("txt",),
    ):
        """
        :param root_dir:                Where the data is stored.
        :param images_dir:              Local path to directory that includes all the images. Path relative to `root_dir`. Can be the same as `labels_dir`.
        :param labels_dir:              Local path to directory that includes all the labels. Path relative to `root_dir`. Can be the same as `images_dir`.
        :param ignore_invalid_labels:   Whether to ignore labels that fail to be parsed. If True ignores and logs a warning, otherwise raise an error.
        :param verbose:                 Whether to show extra information during loading.
        :param image_extensions:        List of image file extensions to load from.
        :param label_extensions:        List of label file extensions to load from.
        """
        self.image_label_tuples = ImageLabelFilesIterator(
            images_dir=os.path.join(root_dir, images_dir),
            labels_dir=os.path.join(root_dir, labels_dir),
            image_extensions=image_extensions,
            label_extensions=label_extensions,
            verbose=verbose,
        )
        self.ignore_invalid_labels = ignore_invalid_labels
        self.verbose = verbose

    def load_image(self, index: int) -> np.ndarray:
        img_file, _ = self.image_label_tuples[index]
        return load_image(path=img_file, channel_format=ImageChannelFormat.RGB)

    def load_labels(self, index: int) -> np.ndarray:
        _, label_path = self.image_label_tuples[index]

        with open(label_path, "r") as file:
            lines = file.readlines()

        labels = []
        for line in filter(lambda x: x != "\n", lines):
            lines_elements = line.split()
            if len(lines_elements) == 5:
                try:
                    labels.append(list(map(float, lines_elements)))
                except ValueError as e:
                    raise ValueError(
                        f"Invalid label: {line} from {label_path}.\nExpected 5 elements (class_id, cx, cy, w, h), got {len(lines_elements)}."
                    ) from e
            else:
                error = f"invalid label: {line} from {label_path}.\n Expected 5 elements (class_id, cx, cy, w, h), got {len(lines_elements)}."
                if self.ignore_invalid_labels:
                    logger.warning(f"Ignoring {error}")
                else:
                    raise RuntimeError(error.capitalize())
        return np.array(labels) if labels else np.zeros((0, 5))

    def __len__(self) -> int:
        return len(self.image_label_tuples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(index)
        labels = self.load_labels(index)
        return image, labels

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(len(self)):
            yield self[i]
