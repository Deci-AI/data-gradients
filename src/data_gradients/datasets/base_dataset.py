import os
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Optional

import numpy as np
from torch.utils.data.dataset import Dataset

from data_gradients.datasets.FolderProcessor import ImageLabelFilesIterator, DEFAULT_IMG_EXTENSIONS
from data_gradients.datasets.utils import load_image_rgb


class BaseImageLabelDirectoryDataset(Dataset, ABC):
    """Base class for any dataset that is primarily made of an image and label directories."""

    def __init__(
        self,
        root_dir: str,
        images_subdir: str,
        labels_subdir: str,
        label_extensions: Sequence[str],
        image_extensions: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        config_path: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        :param root_dir:            Where the data is stored.
        :param images_subdir:       Local path to directory that includes all the images. Path relative to `root_dir`. Can be the same as `labels_subdir`.
        :param labels_subdir:       Local path to directory that includes all the labels. Path relative to `root_dir`. Can be the same as `images_subdir`.
        :param image_extensions:    List of image file extensions to load from.
        :param label_extensions:    List of label file extensions to load from.
        :param config_path:         Path to an optional config file. This config file should contain the list of file ids to include.
                                    If None, all the available images and labels will be loaded.
        :param verbose:             Whether to show extra information during loading.
        """
        config_path = os.path.join(root_dir, config_path) if config_path is not None else None
        self.image_label_tuples = ImageLabelFilesIterator(
            images_dir=os.path.join(root_dir, images_subdir),
            labels_dir=os.path.join(root_dir, labels_subdir),
            config_path=config_path,
            image_extensions=image_extensions,
            label_extensions=label_extensions,
            verbose=verbose,
        )

    def load_image(self, path: str) -> np.ndarray:
        """Load an image from the given path into RGB format."""
        return load_image_rgb(path)

    @abstractmethod
    def load_labels(self, path: str) -> np.ndarray:
        ...

    def __len__(self) -> int:
        return len(self.image_label_tuples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path, labels_path = self.image_label_tuples[index]
        image = self.load_image(path=image_path)
        labels = self.load_labels(path=labels_path)
        return image, labels

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(len(self)):
            yield self[i]
