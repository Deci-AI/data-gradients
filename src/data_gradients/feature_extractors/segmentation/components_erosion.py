import cv2
import numpy as np
import torch

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.utils import class_id_to_name
from data_gradients.batch_processors.preprocessors import contours
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults


@register_feature_extractor()
class ErosionTest(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {"train": {k: 0.0 for k in keys}, "val": {k: 0.0 for k in keys}}
        self._hist_eroded = {
            "train": {k: 0.0 for k in keys},
            "val": {k: 0.0 for k in keys},
        }
        self._kernel = np.ones((3, 3), np.uint8)
        self.ignore_labels = ignore_labels

    def update(self, sample: SegmentationSample):
        for i, image_contours in enumerate(sample.contours):
            eroded_label_tensor = cv2.morphologyEx(sample.mask[i], cv2.MORPH_OPEN, self._kernel)
            eroded_contours = contours.get_contours(eroded_label_tensor)
            for j, cls_contours in enumerate(image_contours):
                if cls_contours:
                    class_id = cls_contours[0].class_id
                    self._hist[sample.split][class_id] += len(cls_contours)
                    if eroded_contours:
                        self._hist_eroded[sample.split][class_id] += len(eroded_contours)

    def _aggregate(self, split: str):
        hist = dict.fromkeys(self._hist[split].keys(), 0.0)
        for cls in self._hist[split]:
            if (self._hist[split][cls]) > 0:
                hist[cls] = np.round(100 * (self._hist_eroded[split][cls] / self._hist[split][cls]), 3)
            else:
                hist[cls] = 0

        hist = class_id_to_name(self.id_to_name, hist)
        values = np.array(list(hist.values()))
        bins = hist.keys()

        results = HistogramResults(
            bin_values=values,
            bin_names=bins,
            title="Erosion & contours comparing",
            x_label="Class",
            y_label="% of disappearing contours after Erosion",
            split=split,
            color=self.colors[split],
            y_ticks=True,
            ax_grid=True,
            plot="bar-plot",
        )
        return results
