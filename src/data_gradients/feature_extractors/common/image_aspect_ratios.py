import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.utils.data_classes.extractor_results import HistogramResults
from data_gradients.feature_extractors.utils import align_histogram_keys


@register_feature_extractor()
class ImagesAspectRatios(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._hist = {"train": dict(), "val": dict()}

    def update(self, sample: ImageSample):
        ar = np.round(sample.image.shape[1] / sample.image.shape[0], 2)
        if ar not in self._hist[sample.split]:
            self._hist[sample.split][ar] = 1
        else:
            self._hist[sample.split][ar] += 1

    def _aggregate(self, split: str):
        self._hist["train"], self._hist["val"] = align_histogram_keys(self._hist["train"], self._hist["val"])

        values = list(self._hist[split].values())
        bins = list(self._hist[split].keys())

        results = HistogramResults(
            bin_names=bins,
            bin_values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title="Image aspect ratios",
            x_label="Aspect ratio [W / H]",
            y_label="# Of Images",
            ticks_rotation=0,
            y_ticks=True,
        )
        return results

    @property
    def description(self):
        return "The distribution of the aspect ratios of the images as a discrete histogram."
