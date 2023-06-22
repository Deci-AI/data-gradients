from abc import ABC, abstractmethod
from typing import Any, List, Optional
import torch


class BaseDatasetExtractor(ABC):
    @abstractmethod
    def is_valid(self, data: Any) -> bool:
        """Check if the data respects the format of this dataset extractor."""
        ...

    @abstractmethod
    def images_extractor(self, data: Any) -> torch.Tensor:
        """Extract images from the data."""
        ...

    @abstractmethod
    def labels_extractor(self, data: Any) -> torch.Tensor:
        """Extract labels from the data."""
        ...


class AutoExtractor:
    def __init__(self, extractors: List[BaseDatasetExtractor]):
        self.extractors = extractors

    def find_extractor(self, data: Any) -> Optional[BaseDatasetExtractor]:
        for extractor in self.extractors:
            if extractor.is_valid(data):
                return extractor
        return None
