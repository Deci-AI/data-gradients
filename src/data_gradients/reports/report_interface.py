from abc import ABC, abstractmethod
from typing import Dict, Any

from data_gradients.feature_extractors.result import FeaturesCollection
from data_gradients.visualize.plot_options import PlotRenderer


class AbstractReportWidget(ABC):
    """
    Abstract class for a single report widget.
    """

    @abstractmethod
    def to_figure(self, results: FeaturesCollection, renderer: PlotRenderer):
        ...

    @abstractmethod
    def to_json(self, results: FeaturesCollection) -> Dict[str, Any]:
        ...
