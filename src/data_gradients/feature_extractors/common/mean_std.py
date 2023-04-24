import numpy as np
import torch

from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils import BatchData
from data_gradients.utils.data_classes.extractor_results import HistoResults


class MeanAndSTD(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()

        self._hist = {"train": {"mean": [], "std": []}, "val": {"mean": [], "std": []}}

    def update(self, data: BatchData):
        for image in data.images:
            self._hist[data.split]["mean"].append(torch.mean(image, [1, 2]))
            self._hist[data.split]["std"].append(torch.std(image, [1, 2]))

    def _aggregate_to_result(self, split: str):
        values, bins = self._aggregate(split)
        results = HistoResults(
            bins=bins,
            values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title="Images mean & std",
            y_label="Mean / STD",
            ticks_rotation=0,
            y_ticks=True,
        )
        return results

    def _aggregate(self, split: str):
        self.merge_dict_splits(self._hist)
        bgr_means = np.zeros(3)
        bgr_std = np.zeros(3)
        for channel in range(3):
            means = [self._hist[split]["mean"][i][channel].item() for i in range(len(self._hist[split]["mean"]))]
            bgr_means[channel] = np.mean(means)
            stds = [self._hist[split]["std"][i][channel].item() for i in range(len(self._hist[split]["std"]))]
            bgr_std[channel] = np.mean(stds)
        values = [
            bgr_means[0],
            bgr_std[0],
            bgr_means[1],
            bgr_std[1],
            bgr_means[2],
            bgr_std[2],
        ]
        bins = [
            "Blue-Mean",
            "Blue-STD",
            "Green-Mean",
            "Green-STD",
            "Red-Mean",
            "Red-STD",
        ]
        return values, bins
