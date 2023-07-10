import numpy as np
import logging
from typing import Sequence, Optional, List, Callable

from data_gradients.datasets.base_dataset import BaseImageLabelDirectoryDataset
from data_gradients.datasets.FolderProcessor import DEFAULT_IMG_EXTENSIONS

logger = logging.getLogger(__name__)


class YoloFormatDetectionDataset(BaseImageLabelDirectoryDataset):
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
        single_line_parser: Optional[Callable[[str], Sequence[float]]] = None,
    ):
        """
        :param root_dir:                Where the data is stored.
        :param images_dir:              Local path to directory that includes all the images. Path relative to `root_dir`. Can be the same as `labels_dir`.
        :param labels_dir:              Local path to directory that includes all the labels. Path relative to `root_dir`. Can be the same as `images_dir`.
        :param ignore_invalid_labels:   Whether to ignore labels that fail to be parsed. If True ignores and logs a warning, otherwise raise an error.
        :param verbose:                 Whether to show extra information during loading.
        :param image_extensions:        List of image file extensions to load from.
        :param label_extensions:        List of label file extensions to load from.
        :param single_line_parser:      Function to parse a single line in the label file.
                                        Should return a list of 5 elements. (class_id and bounding box coordinates)
                                        By default, parse according to the standard Yolo format `<class_id> <cx> <cy> <w> <h>`
        """
        super().__init__(
            root_dir=root_dir,
            images_subdir=images_dir,
            labels_subdir=labels_dir,
            config_path=None,
            verbose=verbose,
            image_extensions=image_extensions,
            label_extensions=label_extensions,
        )
        self.ignore_invalid_labels = ignore_invalid_labels
        self.verbose = verbose
        self.single_line_parser = single_line_parser or parse_yolo_format_line

    def load_labels(self, path: str) -> np.ndarray:
        """Parse a single label file according to `self.single_line_parser`.
        :param path: Local path to the label file.
        :return:     Numpy array representing the labels.
        """

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        labels = []
        for line in filter(lambda x: x != "\n", lines):
            try:
                label = self.single_line_parser(line)
                if label is not None:
                    labels.append(label)
            except Exception as e:
                error = f"invalid label: {line} from {path}. "
                if self.ignore_invalid_labels:
                    logger.warning(f"Ignoring {error}. Exception raise: {e}")
                else:
                    raise RuntimeError(error.capitalize()) from e
        return np.array(labels) if labels else np.zeros((0, 5))


def parse_yolo_format_line(line: str) -> Optional[List[float]]:
    """Parses a line in the standard Yolo format, i.e. `<class_id> <cx> <cy> <w> <h>`. Does not support any variation.
    :param line:    Line representing a bounding box instance. Should be formatted in Yolo format: `<class_id> <cx> <cy> <w> <h>`.
    :return:        List representing the bounding box labels (class_id, cx, cy, w, h).
    """

    elements = line.split()

    # Skip empty lines and comments
    if not elements or elements[0].startswith("#"):
        return None

    if len(elements) != 5:
        raise ValueError(f"Expected at 5 elements per line. Got {len(elements)} elements.")

    # Check if the class_id is a non-negative integer
    class_id = int(elements[0])
    if class_id < 0 or float(elements[0]) != class_id:
        raise ValueError(f"`class_id` should be a non-negative integer. Got `class_id={class_id}`.")

    cx = float(elements[1])
    cy = float(elements[2])
    w = float(elements[3])
    h = float(elements[4])

    return [class_id, cx, cy, w, h]
