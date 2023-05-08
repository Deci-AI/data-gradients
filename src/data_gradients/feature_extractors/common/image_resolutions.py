from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils import BatchData
from data_gradients.utils.data_classes.extractor_results import HistogramResults
from data_gradients.feature_extractors.utils import align_histogram_keys


@register_feature_extractor()
class ImagesResolutions(FeatureExtractorAbstract):
    """
    Extracts the distribution of the image resolutions as a discrete histogram.
    """
    def __init__(self):
        super().__init__()
        self._hist = {"train": dict(), "val": dict()}

    def update(self, data: BatchData):
        for image in data.images:
            res = str(tuple((image.shape[2], image.shape[1])))
            if res not in self._hist[data.split]:
                self._hist[data.split][res] = 1
            else:
                self._hist[data.split][res] += 1

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

    def description(self):
        print("The distribution of the image resolutions as a discrete histogram.")