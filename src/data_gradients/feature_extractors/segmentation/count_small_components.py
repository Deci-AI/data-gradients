from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults
from data_gradients.feature_extractors.utils import normalize_values_to_percentages


@register_feature_extractor()
class CountSmallComponents(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self, minimum_percent_of_an_image):
        super().__init__()
        self._min_size: float = minimum_percent_of_an_image / 100
        self._hist = {
            "train": {f"<{self._min_size}": 0},
            "val": {f"<{self._min_size}": 0},
        }
        self._total_objects = {"train": 0, "val": 0}

    def update(self, sample: SegmentationSample):
        labels_h, labels_w = sample.mask[0].shape
        self._total_objects[sample.split] += sum([len(cls_contours) for cls_contours in sample.contours])
        for class_contours in sample.contours:
            for contour in class_contours:
                self._hist[sample.split][f"<{self._min_size}"] += 1 if contour.area < labels_w * labels_h * self._min_size else 0

    def _aggregate(self, split: str):
        values = normalize_values_to_percentages(self._hist[split].values(), self._total_objects[split])
        bins = list(self._hist[split].keys())

        results = HistogramResults(
            bin_names=bins,
            bin_values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title=f"Components smaller then {self._min_size}% of image",
            x_label="% Components",
            y_label="",
            y_ticks=True,
            ax_grid=True,
        )
        return results
    @property
    def description(self):
        return f"Number of objects in each image smaller then {self._min_size} % of the image size, over all classes."
