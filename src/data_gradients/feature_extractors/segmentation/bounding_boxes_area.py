import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.utils import class_id_to_name
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults


@register_feature_extractor()
class ComponentsSizeDistribution(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()

        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {"train": {k: [] for k in keys}, "val": {k: [] for k in keys}}
        self.ignore_labels = ignore_labels

    def update(self, sample: SegmentationSample):
        image_area = sample.mask.shape[0] * sample.mask.shape[1]
        for class_channel in sample.contours:
            for contour in class_channel:
                self._hist[sample.split][contour.class_id].append(100 * int(contour.bbox_area) / image_area)

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
            title="Components Bounding-Boxes area",
            x_label="Class",
            y_label="Size of BBOX [% of image]",
            ax_grid=True,
            y_ticks=True,
        )
        return results

    @property
    def description(self) -> str:
        return "The distribution of the areas of the boxes that bound connected components of the different classes " \
               "as a histogram.\n" \
               "The size of the objects can significantly affect the performance of your model. If certain classes " \
               "tend to have smaller objects, the model might struggle to segment them, especially if the resolution " \
               "of the images is low "

