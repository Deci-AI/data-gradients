from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple, Dict, Optional, List, Union

import numpy as np
from matplotlib import pyplot as plt

from data_gradients.logging.log_writer import LogWriter
from data_gradients.logging.loggers.results_logger import ResultsLogger
from data_gradients.utils.data_classes.batch_data import BatchData
from data_gradients.utils.data_classes.extractor_results import HistoResults, HeatMapResults


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

        # Logger data
        self.json_object: Dict[str, Optional[ResultsLogger]] = {
            "train": None,
            "val": None,
        }
        self.id_to_name = None

    @abstractmethod
    def update(self, data: BatchData):
        """Accumulate information about samples"""
        raise NotImplementedError

    def aggregate_and_write(self, logger: LogWriter, id_to_name):
        self.id_to_name = id_to_name
        fig, ax = plt.subplots(nrows=self.num_axis[0], ncols=self.num_axis[1], figsize=(10, 5))

        for split in ["train", "val"]:
            results = self.aggregate_to_result(split)
            results.write_plot(ax=ax, fig=fig)
            self.update_json(results=results)

        fig.tight_layout()
        logger.log(title=f"{self.name}/fig", tb_data=fig, json_data=self.json_object)

    @abstractmethod
    def aggregate_to_result(self, split: str) -> HistoResults:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, split: str) -> Tuple[List, List]:
        raise NotImplementedError

    def update_json(self, results: Union[HistoResults, HeatMapResults]):
        keys = results.bins or results.keys
        values = results.json_values or results.values
        self.json_object.update({results.split: dict(zip(keys, values))})

    @staticmethod
    def merge_dict_splits(hist: Dict):
        for key in [*hist["train"], *hist["val"]]:
            if key not in hist["train"]:
                hist["train"][key] = type(hist["val"][key])()
            if key not in hist["val"]:
                hist["val"][key] = type(hist["train"][key])()

        hist["train"] = OrderedDict(sorted(hist["train"].items()))
        hist["val"] = OrderedDict(sorted(hist["val"].items()))

    @staticmethod
    def normalize(values, total):
        if total == 0:
            total = 1
        return [np.round(((100 * value) / total), 3) for value in values]

    @property
    def name(self) -> str:
        return get_title_name(self.__class__.__name__)


class MultiFeatureExtractorAbstract(FeatureExtractorAbstract, ABC):
    def aggregate_and_write(self, logger: LogWriter, id_to_name):
        self.id_to_name = id_to_name

        results_dict = {"train": self.aggregate_to_result("train"), "val": self.aggregate_to_result("val")}

        for key in results_dict["train"].keys():

            fig, ax = plt.subplots(nrows=self.num_axis[0], ncols=self.num_axis[1], figsize=(10, 5))

            for split in ["train", "val"]:
                result = results_dict[split][key]
                result.write_plot(ax=ax, fig=fig)
                self.update_json(results=result)

            fig.tight_layout()
            logger.log(title=f"{self.name}/{key}_{split}/fig", tb_data=fig, json_data=self.json_object)

    @abstractmethod
    def update(self, data: BatchData):
        raise NotImplementedError

    @abstractmethod
    def aggregate_to_result(self, split: str) -> Dict[str, HeatMapResults]:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, split: str) -> Tuple[List, List]:
        raise NotImplementedError


def get_title_name(class_name: str) -> str:
    title_name = class_name[0]
    for char in class_name[1:]:
        if char.isupper():
            title_name += " "
        title_name += char
    return title_name
