from collections import defaultdict
import numpy as np


from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.utils import class_id_to_name
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults


@register_feature_extractor()
class DetectionComponentsSizeDistribution(FeatureExtractorAbstract):
    """
    Detection task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """

    def __init__(self):
        super().__init__()
        self._hist = {"train": defaultdict(list), "val": defaultdict(list)}

    def update(self, sample: DetectionSample):
        image_area = sample.image.shape[0] * sample.image.shape[1]
        for (x1, y1, x2, y2), label in zip(sample.bboxes_xyxy, sample.labels):
            bbox_area = (x2 - x1) * (y2 - y1)
            self._hist[sample.split][label].append(100 * bbox_area / image_area)

    def _aggregate(self, split: str):
        self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
        hist = dict.fromkeys(self._hist[split].keys(), 0.0)
        for cls in self._hist[split]:
            if len(self._hist[split][cls]):
                hist[cls] = float(np.round(np.mean(self._hist[split][cls]), 3))
        values = list(hist.values())
        bins = hist.keys()

        results = HistogramResults(
            bin_names=bins,
            bin_values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title="Bounding-Boxes area",
            x_label="Class",
            y_label="Size of BBOX [% of image]",
            ax_grid=True,
            y_ticks=True,
        )
        return results
