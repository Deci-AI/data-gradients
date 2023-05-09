from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils import BatchData
from data_gradients.utils.data_classes.extractor_results import HistogramResults


@register_feature_extractor()
class NumberOfImagesLabels(FeatureExtractorAbstract):
    """
    Common task feature extractor -
    Count number of images, labels and background images and display as plot-bar.
    """

    def __init__(self):
        super().__init__()
        self._num_images = {"train": 0, "val": 0}
        self._num_labels = {"train": 0, "val": 0}
        self._num_bg_images = {"train": 0, "val": 0}

    def update(self, data: BatchData):
        self._num_images[data.split] += len(data.images)
        self._num_labels[data.split] += len(data.labels)
        for i, label in enumerate(data.labels):
            if label.max() == 0:
                self._num_bg_images[data.split] += 1

    def _aggregate(self, split: str) -> HistogramResults:
        values = [
            self._num_images[split],
            self._num_labels[split],
            self._num_bg_images[split],
        ]
        bins = ["images", "labels", "background images"]

        results = HistogramResults(
            bin_names=bins,
            bin_values=values,
            plot="bar-plot",
            split=split,
            title="# Images & Labels",
            color=self.colors[split],
            y_label="Total #",
            ticks_rotation=0,
            y_ticks=True,
        )
        return results
