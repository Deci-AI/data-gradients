import numpy as np
from typing import Sequence, Optional

from data_gradients.datasets.base_dataset import BaseImageLabelDirectoryDataset
from data_gradients.datasets.FolderProcessor import DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image, ImageChannelFormat


class VOCFormatSegmentationDataset(BaseImageLabelDirectoryDataset):
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
        mask = load_image(path, ImageChannelFormat.RGB)
        classes = np.zeros_like(mask[:, :, 0], dtype=int)

        for class_idx, color in enumerate(self.color_map):
            matching_pixels = np.where(np.all(mask == color, axis=-1))  # Find where in the mask this color appears
            classes[matching_pixels] = class_idx  # Set the class label for these pixels
        return classes
