import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.utils import class_id_to_name
from data_gradients.batch_processors.preprocessors import contours
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults


@register_feature_extractor()
class ComponentsConvexity(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {"train": {k: [] for k in keys}, "val": {k: [] for k in keys}}
        self.ignore_labels = ignore_labels

    def update(self, sample: SegmentationSample):
        for j, cls_contours in enumerate(sample.contours):
            for contour in cls_contours:
                convex_hull = contours.get_convex_hull(contour)
                convex_hull_perimeter = contours.get_contour_perimeter(convex_hull)
                convexity_measure = (contour.perimeter - convex_hull_perimeter) / contour.perimeter
                self._hist[sample.split][contour.class_id].append(convexity_measure)

    def _aggregate(self, split: str):
        hist = dict.fromkeys(self._hist[split].keys(), 0.0)
        for cls in self._hist[split]:
            if len(self._hist[split][cls]):
                hist[cls] = float(np.round(np.mean(self._hist[split][cls]), 3))
        hist = class_id_to_name(self.id_to_name, hist)
        values = np.array(list(hist.values()))
        bins = hist.keys()

        results = HistogramResults(
            bin_values=values,
            bin_names=bins,
            x_label="Class",
            y_label="Convexity measure",
            title="Convexity of components",
            split=split,
            color=self.colors[split],
            y_ticks=True,
            ax_grid=True,
            plot="bar-plot",
        )
        return results

    @property
    def description(self) -> str:
        return "Mean of the convexity measure across all components VS Class ID.\n" \
               "Convexity measure of a component is defined by (" \
               "component_perimeter-convex_hull_perimeter)/convex_hull_perimeter.\n" \
               "High values can imply complex structures which might be difficult to segment."
