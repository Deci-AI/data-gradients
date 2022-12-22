from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class LabelsResolutions(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._res_dict = dict()

    def execute(self, data):
        for label in data.labels:
            res = str(tuple(label.shape[1:]))
            if res not in self._res_dict:
                self._res_dict.update({res: 1})
            else:
                self._res_dict[res] += 1

    def process(self, ax, train):
        create_bar_plot(ax=ax, data=list(self._res_dict.values()), labels=list(self._res_dict.keys()),
                        y_label='# Of Labels', title='Labels resolutions', x_label='Resolution [W, H]',
                        train=train, ticks_rotation=0, color=self.colors[int(train)], yticks=True)
        return self._res_dict


