from abc import ABC, abstractmethod
from typing import Tuple, Dict

from matplotlib import pyplot as plt

from data_gradients.logging.log_writer import LogWriter
from data_gradients.utils.data_classes.batch_data import BatchData
from data_gradients.utils.data_classes.extractor_results import HistogramResults, HeatMapResults


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
        self.colors: Dict[str, str] = {"train": "green", "val": "red"}

        self.id_to_name = None

    def aggregate_and_write(self, logger: LogWriter, id_to_name):
        self.id_to_name = id_to_name
        fig, ax = plt.subplots(nrows=self.num_axis[0], ncols=self.num_axis[1], figsize=(10, 5))

        results_json = {}
        for split in ["train", "val"]:
            results = self._aggregate(split)
            results.write_plot(ax=ax, fig=fig)
            results_json[split] = results.json_values

        fig.tight_layout()
        logger.log(title_name=f"{self.name}/fig", tb_data=fig, json_data=results_json)

    @abstractmethod
    def update(self, data: BatchData):
        """Accumulate information about samples"""
        raise NotImplementedError

    @abstractmethod
    def _aggregate(self, split: str) -> HistogramResults:
        raise NotImplementedError

    @property
    def name(self) -> str:
        class_name = self.__class__.__name__
        title_name = class_name[0]
        for char in class_name[1:]:
            if char.isupper():
                title_name += " "
            title_name += char
        return title_name


class MultiFeatureExtractorAbstract(FeatureExtractorAbstract, ABC):
    def aggregate_and_write(self, logger: LogWriter, id_to_name):
        self.id_to_name = id_to_name

        results_dict = {"train": self._aggregate("train"), "val": self._aggregate("val")}

        for key in results_dict["train"].keys():

            fig, ax = plt.subplots(nrows=self.num_axis[0], ncols=self.num_axis[1], figsize=(10, 5))

            results_json = {}
            for split in ["train", "val"]:
                result = results_dict[split][key]
                result.write_plot(ax=ax, fig=fig)
                results_json[split] = result.json_values

            fig.tight_layout()
            logger.log(title_name=f"{self.name}/{key}_{split}/fig", tb_data=fig, json_data=results_json)

    @abstractmethod
    def update(self, data: BatchData):
        raise NotImplementedError

    @abstractmethod
    def _aggregate(self, split: str) -> Dict[str, HeatMapResults]:
        raise NotImplementedError
