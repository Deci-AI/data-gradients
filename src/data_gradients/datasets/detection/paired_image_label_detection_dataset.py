import os
import numpy as np
import logging
from typing import Tuple, Sequence

from data_gradients.datasets.FolderProcessor import ImageLabelFilesIterator, DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image, ImageChannelFormat


logger = logging.getLogger(__name__)


class PairedImageLabelDetectionDataset:
    """The Paired Image-Label Detection Dataset is a minimalistic and flexible Dataset class for loading datasets
    with a one-to-one correspondence between an image file and a corresponding label text file.

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
    The label files must be structured such that each row represents a bounding box annotation.
    Each bounding box is represented by 5 elements.
      - 1 representing the class id
      - 4 representing the bounding box coordinates.

    The class id can be at the beginning or at the end of the row, but this format needs to be consistent throughout the dataset.
    Example:
      - `class_id x1 y1 x2 y2`
      - `cx, cy, w, h, class_id`
      - `class_id x, y, w, h`
      - ...

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
    from data_gradients.datasets.detection import PairedImageLabelDetectionDataset

    train_loader = PairedImageLabelDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
    val_loader = PairedImageLabelDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
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
        image_extension: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        label_extension: Sequence[str] = ("xml",),
    ):
        """
        :param root_dir:                Where the data is stored.
        :param images_dir:              Local path to directory that includes all the images. Path relative to `root_dir`. Can be the same as `labels_dir`.
        :param labels_dir:              Local path to directory that includes all the labels. Path relative to `root_dir`. Can be the same as `images_dir`.
        :param ignore_invalid_labels:   Whether to ignore labels that fail to be parsed. If True ignores and logs a warning, otherwise raise an error.
        :param verbose:                 Whether to show extra information during loading.
        :param image_extension:         List of image file extensions to load from.
        :param label_extension:         List of label file extensions to load from.
        """
        self.image_label_tuples = ImageLabelFilesIterator(
            images_dir=os.path.join(root_dir, images_dir),
            labels_dir=os.path.join(root_dir, labels_dir),
            image_extension=image_extension or DEFAULT_IMG_EXTENSIONS,
            label_extension=label_extension or [".txt"],
            verbose=verbose,
        )
        self.ignore_invalid_labels = ignore_invalid_labels
        self.verbose = verbose

    def load_image(self, index: int) -> np.ndarray:
        img_file, _ = self.image_label_tuples[index]
        return load_image(path=img_file, channel_format=ImageChannelFormat.RGB)

    def load_annotation(self, index: int) -> np.ndarray:
        _, label_path = self.image_label_tuples[index]

        with open(label_path, "r") as file:
            lines = file.readlines()

        labels = []
        for line in filter(lambda x: x != "\n", lines):
            lines_elements = line.split()
            if len(lines_elements) == 5:
                # Loading everything as floats, even the class_id, because we want to be agnostic of the format.
                labels.append(list(map(float, lines_elements)))
            else:
                error = f"invalid label: {line} from {label_path}.\n Expected 5 elements (class id & 4 bbox coordinates), got {len(lines_elements)}."
                if self.ignore_invalid_labels:
                    logger.warning(f"Ignoring {error}")
                else:
                    raise RuntimeError(error.capitalize())
        return np.array(labels) if labels else np.zeros((0, 5))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(index)
        annotation = self.load_annotation(index)
        return image, annotation
