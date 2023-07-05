import os
import numpy as np
from typing import Tuple, Sequence, Optional


from data_gradients.datasets.FolderProcessor import ImageLabelFilesIterator, ImageLabelConfigIterator, DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image, ImageChannelFormat


class VOCFormatSegmentationDataset:
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
        self.class_names = class_names
        self.color_map = color_map
        if config_path is None:
            self.image_label_tuples = ImageLabelFilesIterator(
                images_dir=os.path.join(root_dir, images_subdir),
                labels_dir=os.path.join(root_dir, labels_subdir),
                image_extensions=image_extensions,
                label_extensions=label_extensions,
                verbose=verbose,
            )
        else:
            self.image_label_tuples = ImageLabelConfigIterator(
                images_dir=os.path.join(root_dir, images_subdir),
                labels_dir=os.path.join(root_dir, labels_subdir),
                config_path=os.path.join(root_dir, config_path),
                image_extensions=image_extensions,
                label_extensions=label_extensions,
                verbose=verbose,
            )

    def __len__(self) -> int:
        return len(self.image_label_tuples)

    def load_image(self, image_path) -> np.ndarray:
        return load_image(image_path, ImageChannelFormat.RGB)

    def load_mask(self, mask_path) -> np.ndarray:
        mask = load_image(mask_path, ImageChannelFormat.RGB)
        classes = np.zeros_like(mask[:, :, 0], dtype=int)

        for class_idx, color in enumerate(self.color_map):
            matching_pixels = np.where(np.all(mask == color, axis=-1))  # Find where in the mask this color appears
            classes[matching_pixels] = class_idx  # Set the class label for these pixels

        return classes

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path, label_path = self.image_label_tuples[index]
        img = self.load_image(image_path=image_path)
        label = self.load_mask(mask_path=label_path)
        return img, label

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(len(self)):
            yield self[i]
