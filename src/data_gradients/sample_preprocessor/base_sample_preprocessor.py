from abc import ABC, abstractmethod
from typing import Iterator, Iterable

from data_gradients.utils.data_classes import ImageSample
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType
from data_gradients.dataset_adapters.config.data_config import DataConfig


class AbstractSamplePreprocessor(ABC):
    """Abstract class responsible for pre-processing dataset/dataloader output into a known sample format.

    This class it the connector between users data and the feature extracting processes as it ensures that
    all the data passed to these feature extractor are of the same known format.

    :attr data_config: Configuration of the data adapter.
    """

    def __init__(self, data_config: DataConfig):
        """
        :param data_config: Configuration of the data adapter.
        """
        self.data_config = data_config

    @abstractmethod
    def preprocess_samples(self, dataset: Iterable[SupportedDataType], split: str) -> Iterator[ImageSample]:
        """Pre-process the output of a dataset/dataloader into a known sample format.
        :param dataset: Dataset/dataloader to be processed.
        :param split:   Split of the dataset/dataloader. ("train", "val", ...)
        :returns:       Iterator yielding the processed samples of the dataset/dataloader one by one.
        """
        ...
