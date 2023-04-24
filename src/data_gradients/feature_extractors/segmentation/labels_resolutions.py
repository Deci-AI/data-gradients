from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils import SegBatchData
from data_gradients.utils.data_classes.extractor_results import HistoResults


class LabelsResolutions(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._hist = {"train": dict(), "val": dict()}

    def update(self, data: SegBatchData):
        for label in data.labels:
            res = str(tuple((label.shape[2], label.shape[1])))
            if res not in self._hist[data.split]:
                self._hist[data.split].update({res: 1})
            else:
                self._hist[data.split][res] += 1

    def _aggregate_to_result(self, split: str):
        values, bins = self._aggregate(split)
        results = HistoResults(
            bins=bins,
            values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title="Labels resolutions",
            x_label="Resolution [W, H]",
            y_label="# Of Labels",
            ticks_rotation=0,
            y_ticks=True,
        )
        return results

    def _aggregate(self, split: str):
        self.merge_dict_splits(self._hist)
        values = list(self._hist[split].values())
        bins = list(self._hist[split].keys())
        return values, bins
