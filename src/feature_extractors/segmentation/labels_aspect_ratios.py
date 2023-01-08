import numpy as np

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.utils import SegBatchData
from src.utils.data_classes import Results


class LabelsAspectRatios(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._hist = {'train': dict(), 'val': dict()}
        self._channels_last = False

    def _execute(self, data: SegBatchData):
        for label in data.labels:
            ar = np.round(label.shape[2] / label.shape[1], 2)
            if ar not in self._hist[data.split]:
                self._hist[data.split][ar] = 1
            else:
                self._hist[data.split][ar] += 1

    def _post_process(self, split):
        values, bins = self._process_data(split)
        results = Results(bins=bins,
                          values=values,
                          plot='bar-plot',
                          split=split,
                          color=self.colors[split],
                          title='Labels aspect ratios',
                          x_label='Aspect ratio [W / H]',
                          y_label='# Of Labels',
                          ticks_rotation=0,
                          y_ticks=True
                          )
        return results

    def _process_data(self, split: str):
        self.merge_dict_splits(self._hist)
        values = list(self._hist[split].values())
        bins = list(self._hist[split].keys())
        return values, bins
