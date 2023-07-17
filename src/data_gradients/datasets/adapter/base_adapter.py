from data_gradients.utils.data_classes.data_samples import ImageSample

from abc import ABC, abstractmethod
from typing import List, Optional, Iterable, Sized

from data_gradients.batch_processors.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.batch_processors.formatters.base import BatchFormatter
from data_gradients.config.data.data_config import DataConfig


class BaseDatasetAdapter(ABC):
    def __init__(
        self,
        data_iterable: Iterable,
        dataset_output_mapper: DatasetOutputMapper,
        formatter: BatchFormatter,
        data_config: Optional[DataConfig] = None,
    ):
        self.data_iterable = data_iterable
        self.data_config = data_config

        self.dataset_output_mapper = dataset_output_mapper
        self.formatter = formatter

    @staticmethod
    def resolve_class_names(class_names: List[str], n_classes: int) -> List[str]:
        # Check values of `n_classes` and `class_names` to define `class_names`.
        if n_classes and class_names:
            raise RuntimeError("`class_names` and `n_classes` cannot be specified at the same time")
        elif n_classes is None and class_names is None:
            raise RuntimeError("Either `class_names` or `n_classes` must be specified")
        return class_names or list(map(str, range(n_classes)))

    @staticmethod
    def resolve_class_names_to_use(class_names: List[str], class_names_to_use: List[str]) -> List[str]:
        # Define `class_names_to_use`
        if class_names_to_use:
            invalid_class_names_to_use = set(class_names_to_use) - set(class_names)
            if invalid_class_names_to_use != set():
                raise RuntimeError(f"You defined `class_names_to_use` with classes that are not listed in `class_names`: {invalid_class_names_to_use}")
        return class_names_to_use or class_names

    def __len__(self) -> int:
        """Length of the dataset if available. Otherwise, None."""
        return len(self.data_iterable) if isinstance(self.data_iterable, Sized) else None

    def __iter__(self):
        for data in self.data_iterable:
            # data can be a batch or a sample
            images, labels = self.dataset_output_mapper.extract(data)
            images, labels = self.formatter.format(images, labels)
            yield images, labels

    def close(self):
        self.data_config.close()

    @abstractmethod
    def samples_iterator(self, split_name: str) -> Iterable[ImageSample]:
        ...
