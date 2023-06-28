import os
import numpy as np
import logging
from typing import List, Optional, Tuple

from data_gradients.datasets.FolderProcessor import ImageLabelFilesIterator, DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image, ImageChannelFormat


logger = logging.getLogger(__name__)


class YoloDetectionDataset:
    """YoloDataset is a minimalistic and flexible Dataset class for loading YoloV5 and Darknet format datasets.
    YoloDataset supports the following dataset structures:

    Example 1:
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

    Example 2 (same directory for images and labels):
        dataset_root/
            ├── train/
            │   ├── 1.jpg
            │   ├── 1.txt
            │   ├── 2.jpg
            │   ├── 2.txt
            │   └── ...
            └── validation/
                ├── ...
    """

    def __init__(
        self,
        root_dir: str,
        images_dir: str,
        labels_dir: str,
        ignore_invalid_labels: bool = True,
        verbose: bool = False,
        image_extension: Optional[List[str]] = None,
        label_extension: Optional[List[str]] = None,
    ):
        """
        :param root_dir:                Where the data is stored.
        :param images_dir:              Local path to directory that includes all the images. Path relative to `root_dir`. Can be the same as `labels_dir`.
        :param labels_dir:              Local path to directory that includes all the labels. Path relative to `root_dir`. Can be the same as `images_dir`.
        :param ignore_invalid_labels:   Whether to ignore labels that fail to be parsed. If True ignores and logs a warning, otherwise raise an error.
        :param verbose:                 Whether to show extra information during loading.
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

        if not os.path.exists(label_path):
            return np.zeros((0, 5))

        with open(label_path, "r") as file:
            lines = file.readlines()

        labels_yolo_format = []
        for line in filter(lambda x: x != "\n", lines):
            try:
                label_id, cx, cw, w, h = line.split()
                labels_yolo_format.append([int(label_id), float(cx), float(cw), float(w), float(h)])
            except Exception as e:
                if self.ignore_invalid_labels:
                    if self.verbose:
                        logger.warning(
                            f"Line `{line}` of file {label_path} will be ignored because could not parsed into (cls_id, cx, cy, w, h) format.\n"
                            f"Got Exception: {e}"
                        )
                else:
                    raise e
        return np.array(labels_yolo_format) if labels_yolo_format else np.zeros((0, 5))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(index)
        annotation = self.load_annotation(index)
        return image, annotation
