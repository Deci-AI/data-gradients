import numpy as np

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot, create_json_object


class LabelsAspectRatios(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._hist = {'train': dict(), 'val': dict()}
        self._channels_last = False

    def execute(self, data):
        for label in data.labels:
            ar = np.round(label.shape[2] / label.shape[1], 2)
            if ar not in self._hist[data.split]:
                self._hist[data.split][ar] = 1
            else:
                self._hist[data.split][ar] += 1

    def _process(self):
        self.merge_dict_splits(self._hist)
        for split in ['train', 'val']:
            create_bar_plot(ax=self.ax, data=list(self._hist[split].values()),
                            labels=list(self._hist[split].keys()), y_label='# Of Labels',
                            title='Labels aspect ratios', x_label='Aspect ratio [W / H]', split=split, ticks_rotation=0,
                            color=self.colors[split], yticks=True)
            self.json_object.update({split: self._hist[split]})
