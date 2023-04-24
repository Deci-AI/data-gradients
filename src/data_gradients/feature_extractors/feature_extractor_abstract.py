from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple, Dict, Optional, Union

import numpy as np
from matplotlib import pyplot as plt

from data_gradients.logging.logger import Logger
from data_gradients.logging.logger_utils import (
    create_json_object,
    write_bar_plot,
    write_heatmap_plot,
)
from data_gradients.logging.results_logger import ResultsLogger
from data_gradients.utils.data_classes.batch_data import BatchData
from data_gradients.utils.data_classes.extractor_results import VisualizationResults, HistogramResults, HeatMapResults


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

    @abstractmethod
    def _aggregate(self, split: str) -> VisualizationResults:
        raise NotImplementedError

    def aggregate_and_write(self, logger: Logger, id_to_name):
        self.id_to_name = id_to_name
        self.fig, ax = plt.subplots(*self.num_axis, figsize=(10, 5))

        for split in ["train", "val"]:
            results = self._aggregate(split)
            self.update_json(results, ax)

        self.fig.tight_layout()
        title_name = logger.get_title_name(self.__class__.__name__) + "/fig"
        logger.log(title_name=title_name, tb_data=self.fig, json_data=self.json_object)

    def update_json(self, results: Union[HistogramResults, HeatMapResults], ax):
        if results.plot == "bar-plot":
            write_bar_plot(ax=ax, results=results)
        elif results.plot == "heat-map":
            write_heatmap_plot(ax=ax[int(results.split != "train")], results=results, fig=self.fig)
        else:
            raise NotImplementedError(
                f"Got plot key {results.plot}\
             while only supported plots are ['bar-plot', 'heat-map']"
            )

        json_obj = create_json_object(
            results.json_values if results.json_values else results.values,
            results.bins if results.bins else results.keys,
        )
        self.json_object.update({results.split: json_obj})

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


class MultiClassProcess(FeatureExtractorAbstract):
    def aggregate_and_write(self, logger: Logger, id_to_name):
        self.id_to_name = id_to_name

        results = dict.fromkeys(["train", "val"])
        for split in results:
            results[split] = self._aggregate(split)

        for key in results["train"].keys():

            self.fig, ax = plt.subplots(*self.num_axis, figsize=(10, 5))

            for split in ["train", "val"]:
                self.update_json(results[split][key], ax)

            self.fig.tight_layout()

            title_name = f"{logger.get_title_name(self.__class__.__name__)}/{key}_{split}/fig"
            logger.log(title_name=title_name, tb_data=self.fig, json_data=self.json_object)

    @abstractmethod
    def update(self, data: BatchData):
        raise NotImplementedError

    @abstractmethod
    def _aggregate(self, split: str) -> Dict[str, HeatMapResults]:
        raise NotImplementedError
