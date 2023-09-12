from abc import ABC
from typing import List, Iterable, Sized, Tuple, Optional

import torch

from data_gradients.config.data.typing import SupportedDataType
from data_gradients.config.data.data_config import DataConfig

from data_gradients.dataset_adapters.formatters.base import BatchFormatter
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper


class BaseDatasetAdapter(ABC):
    """Wrap a dataset and applies transformations on data points.
    It acts as a base class for specific dataset adapters that cater to specific data structures.

    :param formatter:       Instance of BatchFormatter that is used to validate and format the batches of images and labels
                            into the appropriate format for a given task.
    :param data_config:     Instance of DataConfig class that manages dataset/dataloader configurations.
    :param data_iterable:   (Optional) Iterable object that yields data points from the dataset. If None, `BaseDatasetAdapter` won't be iterable.
    """

    def __init__(
        self,
        dataset_output_mapper: DatasetOutputMapper,
        formatter: BatchFormatter,
        data_config: DataConfig,
        class_names: List[str],
        data_iterable: Optional[Iterable[SupportedDataType]] = None,
    ):
        self.data_iterable = data_iterable
        self.data_config = data_config

        self.dataset_output_mapper = dataset_output_mapper
        self.formatter = formatter
        self.class_names = class_names

    @staticmethod
    def resolve_class_names(class_names: List[str], n_classes: int) -> List[str]:
        """Ensure that either `class_names` or `n_classes` is specified, but not both. Return the list of class names that will be used."""
        if n_classes and class_names:
            raise RuntimeError("`class_names` and `n_classes` cannot be specified at the same time")
        elif n_classes is None and class_names is None:
            raise RuntimeError("Either `class_names` or `n_classes` must be specified")
        return class_names or list(map(str, range(n_classes)))

    @staticmethod
    def resolve_class_names_to_use(class_names: List[str], class_names_to_use: List[str]) -> List[str]:
        """Define `class_names_to_use` from `class_names` if it is specified. Otherwise, return the list of class names that will be used."""
        if class_names_to_use:
            invalid_class_names_to_use = set(class_names_to_use) - set(class_names)
            if invalid_class_names_to_use != set():
                raise RuntimeError(f"You defined `class_names_to_use` with classes that are not listed in `class_names`: {invalid_class_names_to_use}")
        return class_names_to_use or class_names

    def close(self):
        """Run any action required to cleanly close the object. May include saving cache."""
        self.data_config.close()

    def __len__(self) -> int:
        """Length of the dataset if available. Otherwise, None."""
        if self.data_iterable is None:
            raise ValueError(f"Impossible to get len(`{self.__class__.__name__}`) because `data_iterable` was not initialized.")
        return len(self.data_iterable) if isinstance(self.data_iterable, Sized) else None

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over the dataset and return a batch of images and labels."""
        if self.data_iterable is None:
            raise ValueError(f"Impossible to iterate over `{self.__class__.__name__}` because `data_iterable` was not initialized.")

        for data in self.data_iterable:
            # data can be a batch or a sample
            images, labels = self.adapt_batch(data)
            yield images, labels

    def adapt_batch(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterate over the dataset and return a batch of images and labels."""
        # data can be a batch or a sample
        images, labels = self.dataset_output_mapper.extract(data)
        images, labels = self.formatter.format(images, labels)
        return images, labels

    def adapt_iterable(self, data_iterable: Iterable) -> Iterable:
        for item in data_iterable:
            yield self.adapt_batch(item)
