from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.visualize.plot_options import CommonPlotOptions


@dataclass
class Feature:
    """Feature extracted from the whole dataset."""

    data: Union[pd.DataFrame, np.ndarray, plt.Figure]
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

    @property
    def notice(self) -> Optional[str]:
        return None

    @property
    def warning(self) -> Optional[str]:
        return None

    def setup_data_sources(self, tran_data: Iterable, val_data: Iterable):
        """
        Called in AnalysisManagerAbstract.__init__ for the purpose of exposing tran_data and val_data
         to the feature in case some information is needed.
        """
        pass

    def __repr__(self):
        return self.__class__.__name__
