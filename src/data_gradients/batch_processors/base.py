from abc import ABC
from typing import Mapping, Union, List, Tuple, Iterable

from data_gradients.utils.data_classes import ImageSample
from data_gradients.batch_processors.adapters.dataset_adapter import DatasetAdapter
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor
from data_gradients.batch_processors.formatters.base import BatchFormatter


class BatchProcessor(ABC):
    """A callable abstract base class responsible for processing raw batches (coming from a DataLoader) into
    ready-to-analyze batch objects. It handles extraction, validation, and preprocessing of the images and labels.

    """

    def __init__(self, dataset_adapter: DatasetAdapter, batch_formatter: BatchFormatter, batch_preprocessor: BatchPreprocessor):
        """
        :param dataset_adapter:     Object responsible for extracting images and labels from the raw batch.
        :param batch_formatter:     Object responsible for validating the format of images and labels.
        :param batch_preprocessor:  Object responsible for preprocessing images and labels into a ready-to-analyze batch object.
        """
        self.dataset_adapter = dataset_adapter
        self.batch_formatter = batch_formatter
        self.batch_preprocessor = batch_preprocessor

    def process(self, unprocessed_batch: Union[Tuple, List, Mapping], split: str) -> Iterable[ImageSample]:
        images, labels = self.dataset_adapter.extract(unprocessed_batch)
        images, labels = self.batch_formatter.format(images, labels)
        for sample in self.batch_preprocessor.preprocess(images, labels, split):
            yield sample
