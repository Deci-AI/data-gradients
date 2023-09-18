from abc import ABC
from typing import List, Tuple

import torch

from data_gradients.config.data.data_config import DataConfig

from data_gradients.dataset_adapters.formatters.base import BatchFormatter
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.config.data.typing import SupportedDataType


class BaseDatasetAdapter(ABC):
    """Wrap a dataset and applies transformations on data points.
    It acts as a base class for specific dataset adapters that cater to specific data structures.

    :param formatter:       Instance of BatchFormatter that is used to validate and format the batches of images and labels
                            into the appropriate format for a given task.
    :param data_config:     Instance of DataConfig class that manages dataset/dataloader configurations.
    """

    def __init__(
        self,
        dataset_output_mapper: DatasetOutputMapper,
        formatter: BatchFormatter,
        data_config: DataConfig,
        class_names: List[str],
    ):
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

    def adapt(self, data: SupportedDataType) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapt an input data (Batch or Sample) into a standard format.

        :param data:     Input data to be adapted.
                            - Can represent a batch or a sample.
                            - Can be structured in a wide range of formats. (list, dict, ...)
                            - Can be formatted in a wide range of formats. (image: HWC, CHW, ... - label: label_cxcywh, xyxy_label, ...)
        :return:         Tuple of images and labels.
                            - Image will be formatted to (BS, H, W, C) - BS = 1 if original data is a single sample
                            - Label will be formatted to a standard format that depends on the task.
        """
        images, labels = self.dataset_output_mapper.extract(data)
        images, labels = self.formatter.format(images, labels)
        return images, labels
