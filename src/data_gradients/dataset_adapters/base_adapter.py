from abc import ABC
from typing import Tuple, List

import torch

from data_gradients.dataset_adapters.config.data_config import DataConfig

from data_gradients.dataset_adapters.formatters.base import BatchFormatter
from data_gradients.dataset_adapters.output_mapper.dataset_output_mapper import DatasetOutputMapper
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType
from data_gradients.utils.data_classes.data_samples import Image


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
    ):
        self.data_config = data_config

        self.dataset_output_mapper = dataset_output_mapper
        self.formatter = formatter

    def adapt(self, data: SupportedDataType) -> Tuple[List[Image], torch.Tensor]:
        """Adapt an input data (Batch or Sample) into a standard format.

        :param data:     Input data to be adapted.
                            - Can represent a batch or a sample.
                            - Can be structured in a wide range of formats. (list, dict, ...)
                            - Can be formatted in a wide range of formats. (image: HWC, CHW, ... - label: label_cxcywh, xyxy_label, ...)
        :return:         Tuple of images and labels.
                            - List of Image
                            - Label will be formatted to a standard format that depends on the task.
        """
        images, labels = self.dataset_output_mapper.extract(data)
        images, labels = self.formatter.format(images, labels)
        return images, labels
