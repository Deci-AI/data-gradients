from collections import defaultdict

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes import ImageSample
from data_gradients.utils.data_classes.extractor_results import HistogramResults
from data_gradients.feature_extractors.utils import align_histogram_keys


@register_feature_extractor()
class ImagesResolutions(FeatureExtractorAbstract):
    """
    Extracts the distribution of the image resolutions as a discrete histogram.
    """

    def __init__(self):
        super().__init__()
        self._hist = defaultdict(dict)

    def update(self, sample: ImageSample):
        rows, cols = sample.image.shape[:2]
        res = str(tuple((cols, rows)))
        if res not in self._hist[sample.split]:
            self._hist[sample.split][res] = 1
        else:
            self._hist[sample.split][res] += 1

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
            title="Image resolutions",
            x_label="Resolution [W, H]",
            y_label="# Of Images",
            ticks_rotation=0,
            y_ticks=True,
        )
        return results

    @property
    def description(self) -> str:
        return "The distribution of the image resolutions as a discrete histogram. \n Note that if images are " \
               "rescaled or padded, this plot will show the size after rescaling and padding. "
