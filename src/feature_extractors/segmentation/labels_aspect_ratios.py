import numpy as np

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class LabelsAspectRatios(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._ar_dict = dict()
        self._channels_last = False

    def execute(self, data):
        for label in data.labels:
            ar = np.round(label.shape[1] / label.shape[2], 2)
            if ar not in self._ar_dict:
                self._ar_dict[ar] = 1
            else:
                self._ar_dict[ar] += 1

    def process(self, ax, train):
        create_bar_plot(ax=ax, data=list(self._ar_dict.values()), labels=list(self._ar_dict.keys()),
                        y_label='# Of Labels', title='Labels aspect ratios', x_label='Aspect ratio [W / H]',
                        train=train, ticks_rotation=0, color=self.colors[int(train)], yticks=True)
        return self._ar_dict
