import numpy as np
from typing import Sequence, Optional

from data_gradients.datasets.base_dataset import BaseImageLabelDirectoryDataset
from data_gradients.datasets.FolderProcessor import DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image_rgb


class VOCFormatSegmentationDataset(BaseImageLabelDirectoryDataset):
    """The VOC format Segmentation Dataset supports datasets where labels are stored as images, with each color in the image representing a different class.

    #### Expected folder structure
    Similar to the VOCFormatDetectionDataset, this class also expects certain folder structures. For example:

    Example: Separate directories for images and labels
    ```
        dataset_root/
            ├── train.txt
            ├── validation.txt
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
                │   ├── 1.png
                │   ├── 2.png
                │   └── ...
                ├── test/
                │   ├── ...
                └── validation/
                    ├── ...
    ```
    Each label image should be a color image where the color of each pixel corresponds to the class of that pixel.

    The (optional) config file should include the list image ids to include.
    ```
    1
    5
    6
    # And so on ...
    ```
    The associated images/labels will then be loaded from the images_subdir and labels_subdir.
    If config_path is not provided, all images will be used.

    #### Instantiation
    ```
    from data_gradients.datasets.segmentation import VOCFormatSegmentationDataset

    color_map = [
        [0, 0, 0],      # class 0
        [255, 0, 0],    # class 1
        [0, 255, 0],    # class 2
        # ...
    ]

    train_set = VOCFormatSegmentationDataset(
        root_dir="<path/to/dataset_root>",
        images_subdir="images/train",
        labels_subdir="labels/train",
        class_names=["background", "class1", "class2"],
        color_map=color_map,
        config_path="train.txt"
    )
    val_set = VOCFormatSegmentationDataset(
        root_dir="<path/to/dataset_root>",
        images_subdir="images/validation",
        labels_subdir="labels/validation",
        class_names=["background", "class1", "class2"],
        color_map=color_map,
        config_path="validation.txt"
    )
    ```
    """

    def __init__(
        self,
        root_dir: str,
        images_subdir: str,
        labels_subdir: str,
        class_names: Sequence[str],
        color_map: Sequence[Sequence[int]],
        config_path: Optional[str] = None,
        verbose: bool = False,
        image_extensions: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        label_extensions: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
    ):
        """
        :param root_dir:            Where the data is stored.
        :param images_subdir:       Local path to directory that includes all the images. Path relative to `root_dir`.
        :param labels_subdir:       Local path to directory that includes all the labels. Path relative to `root_dir`.
        :param class_names:         List of class names. This is required to be able to parse the class names into class ids.
        :param color_map:           List of RGB colors associated with each class.
                                    The color of each pixel in the label images will be compared to this list to determine the class of the pixel.
        :param config_path:         Path to an optional config file. This config file should contain the list of file ids to include.
                                    If None, all the available images and labels will be loaded.
        :param verbose:             Whether to show extra information during loading.
        :param image_extensions:    List of image file extensions to load from.
        :param label_extensions:    List of label file extensions to load from.
        """
        super().__init__(
            root_dir=root_dir,
            images_subdir=images_subdir,
            labels_subdir=labels_subdir,
            config_path=config_path,
            verbose=verbose,
            image_extensions=image_extensions,
            label_extensions=label_extensions,
        )
        self.class_names = class_names
        self.color_map = color_map

    def load_labels(self, path: str) -> np.ndarray:
        """Load a label image and convert it into a 2D array where each value represents the class of the pixel.

        :param path:    Path to the label image file.
        :return:        Mask of the label image, in (H, W) format, where Mij represents the class_id.
        """
        mask = load_image_rgb(path)
        classes = np.zeros_like(mask[:, :, 0], dtype=int)

        for class_idx, color in enumerate(self.color_map):
            matching_pixels = np.where(np.all(mask == color, axis=-1))  # Find where in the mask this color appears
            classes[matching_pixels] = class_idx  # Set the class label for these pixels
        return classes
