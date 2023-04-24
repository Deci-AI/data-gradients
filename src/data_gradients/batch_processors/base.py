from abc import ABC
from typing import Mapping, Union, List, Tuple

from data_gradients.utils import BatchData
from data_gradients.batch_processors.batch_extractor import BatchExtractor
from data_gradients.batch_processors.preprocessors.base import Preprocessor
from data_gradients.batch_processors.validators.base import BatchValidator


class BatchProcessor(ABC):
    def __init__(self, batch_extractor: BatchExtractor, batch_validator: BatchValidator, batch_preprocessor: Preprocessor):
        self.batch_extractor = batch_extractor
        self.batch_validator = batch_validator
        self.batch_preprocessor = batch_preprocessor

    def __call__(self, unprocessed_batch: Union[Tuple, List, Mapping]) -> BatchData:
        images, labels = self.batch_extractor(unprocessed_batch)
        images, labels = self.batch_validator(images, labels)
        batch = self.batch_preprocessor(images, labels)
        return batch

    @property
    def images_route(self) -> List[str]:
        return self.batch_extractor.images_route

    @property
    def labels_route(self) -> List[str]:
        return self.batch_extractor.labels_route
