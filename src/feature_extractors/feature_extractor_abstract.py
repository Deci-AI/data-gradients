from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple, Dict, Optional

from matplotlib import pyplot as plt

from src.logger.results_logger import ResultsLogger
from src.utils import BatchData


class FeatureExtractorAbstract(ABC):
    """
    Main feature extractors class.
    Mandatory method to implement are execute() and process().
        * execute method will get a batch of data and will iterate over it to retrieve the features out of the data.
        * process will have some calculations over the features extracted in order to make it ready for histogram,
          plotting, etc.
    :param: colors - determine graph color for train/val bars
    :param: single_axis - determine if a graph should be combined axis train/val bars,
                          or have a separate axis for each of them.
    """
    def __init__(self):
        self.num_axis: Tuple[int, int] = (1, 1)
        self.colors: Dict[str, str] = {'train': 'green',
                                       'val': 'red'}

        # Logger data
        self.fig = None
        self.ax = None
        self.json_object: Dict[str, Optional[ResultsLogger]] = {'train': None, 'val': None}
        self.id_to_name = None

    def execute(self, data: BatchData):
        self._execute(data)

    @abstractmethod
    def _execute(self, data: BatchData):
        raise NotImplementedError

    def process(self, loggers: Dict[str, ResultsLogger], id_to_name):
        self.id_to_name = id_to_name

        self.fig, self.ax = plt.subplots(*self.num_axis, figsize=(10, 5))

        self._process()

        self.fig.tight_layout()
        self.log(logger=loggers['TB'],
                 title=self.__class__.__name__,
                 data=self.fig)

        self.log(logger=loggers['JSON'],
                 title=self.__class__.__name__,
                 data=self.json_object)

    @abstractmethod
    def _process(self):
        pass

    @staticmethod
    def log(logger: ResultsLogger, title: str, data):
        logger.log(title, data)

    @staticmethod
    def merge_dict_splits(hist: Dict):
        for key in [*hist['train'], *hist['val']]:
            for split in [*hist]:
                if key not in list(hist[split].keys()):
                    hist[split][key] = 0.
                hist[split] = OrderedDict(sorted(hist[split].items()))
