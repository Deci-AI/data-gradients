from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils import BatchData
from data_gradients.utils.data_classes.extractor_results import HistoResults


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

    def aggregate_to_result(self, split: str):
        values, bins = self.aggregate(split)
        results = HistoResults(
            bins=bins,
            values=values,
            plot="bar-plot",
            split=split,
            title="# Images & Labels",
            color=self.colors[split],
            y_label="Total #",
            ticks_rotation=0,
            y_ticks=True,
        )
        return results

    def aggregate(self, split: str):
        values = [
            self._num_images[split],
            self._num_labels[split],
            self._num_bg_images[split],
        ]
        bins = ["images", "labels", "background images"]
        return values, bins
