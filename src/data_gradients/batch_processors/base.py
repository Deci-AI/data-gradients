from abc import ABC
from typing import Mapping, Union, List, Tuple

from data_gradients.utils import BatchData
from data_gradients.batch_processors.extractors.batch_extractor import BatchExtractor
from data_gradients.batch_processors.preprocessors.base import BatchPreprocessor
from data_gradients.batch_processors.formatters.base import BatchFormatter


class BatchProcessor(ABC):
    """A callable abstract base class responsible for processing raw batches (coming from a DataLoader) into
    ready-to-analyze batch objects. It handles extraction, validation, and preprocessing of the images and labels.

    """

    def __init__(self, batch_extractor: BatchExtractor, batch_formatter: BatchFormatter, batch_preprocessor: BatchPreprocessor):
        """
        :param batch_extractor:     Object responsible for extracting images and labels from the raw batch.
        :param batch_formatter:     Object responsible for validating the format of images and labels.
        :param batch_preprocessor:  Object responsible for preprocessing images and labels into a ready-to-analyze batch object.
        """
        self.batch_extractor = batch_extractor
        self.batch_formatter = batch_formatter
        self.batch_preprocessor = batch_preprocessor

    def process(self, unprocessed_batch: Union[Tuple, List, Mapping], split: str) -> BatchData:
        images, labels = self.batch_extractor.extract(unprocessed_batch)
        images, labels = self.batch_formatter.format(images, labels)
        batch = self.batch_preprocessor.preprocess(images, labels)
        batch.split = split
        return batch

    @property
    def images_route(self) -> List[str]:
        return self.batch_extractor.images_route

    @property
    def labels_route(self) -> List[str]:
        return self.batch_extractor.labels_route
