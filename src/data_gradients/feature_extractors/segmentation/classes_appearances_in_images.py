import numpy as np
import torch

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.utils import class_id_to_name
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults
from data_gradients.feature_extractors.utils import normalize_values_to_percentages


@register_feature_extractor()
class AppearancesInImages(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    For each class, calculate percentage of images it appears in out of all images in set.
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        self.ignore_labels = ignore_labels
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {"train": dict.fromkeys(keys, 0), "val": dict.fromkeys(keys, 0)}
        self._number_of_images = {"train": 0, "val": 0}

    def update(self, sample: SegmentationSample):
        self._number_of_images[sample.split] += 1
        for j, class_channel in enumerate(sample.mask):
            if np.any(class_channel) > 0:
                self._hist[sample.split][j] += 1

    def _aggregate(self, split: str):
        self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
        values = normalize_values_to_percentages(self._hist[split].values(), self._number_of_images[split])
        bins = self._hist[split].keys()

        results = HistogramResults(
            bin_names=bins,
            bin_values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title="% Images that class appears in",
            x_label="Class #",
            y_label="Images appeared in [%]",
            y_ticks=True,
            ax_grid=True,
        )
        return results

    @property
    def description(self):
        return "Percentage of images containing an appearance from VS class. \n" \
               "If a certain class has significantly fewer instances, the model might not learn to recognize it " \
               "effectively, which can lead to poor performance when predicting this class. It is recommended to use " \
               "this feature along with the PixelsPerClass feature extractor."
