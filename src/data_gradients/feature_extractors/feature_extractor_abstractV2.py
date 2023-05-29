from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.visualize.plot_options import CommonPlotOptions


@dataclass
class Feature:
    """Feature extracted from the whole dataset."""

    data: Union[pd.DataFrame, np.ndarray]
    plot_options: CommonPlotOptions

    json: Union[dict, list]


class AbstractFeatureExtractor(ABC):
    @abstractmethod
    def update(self, sample: ImageSample):
        """Accumulate information about samples"""
        raise NotImplementedError()

    @abstractmethod
    def aggregate(self) -> Feature:
        raise NotImplementedError()

    @property
    def description(self) -> str:
        raise NotImplementedError()

    @property
    def title(self) -> str:
        raise NotImplementedError()
