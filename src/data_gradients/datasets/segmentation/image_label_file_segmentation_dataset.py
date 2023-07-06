import numpy as np
from typing import Sequence, Optional

from data_gradients.datasets.base_dataset import BaseImageLabelDirectoryDataset
from data_gradients.datasets.FolderProcessor import DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image, ImageChannelFormat


class ImageLabelFileIteratorSegmentationDataset(BaseImageLabelDirectoryDataset):
    def __init__(
        self,
        root_dir: str,
        images_subdir: str,
        labels_subdir: str,
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

    def load_labels(self, path: str) -> np.ndarray:
        return load_image(path, ImageChannelFormat.RGB)
