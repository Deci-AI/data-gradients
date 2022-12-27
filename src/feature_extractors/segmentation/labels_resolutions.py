from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class LabelsResolutions(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._res_dict = {'train': dict(), 'val': dict()}

    def _execute(self, data):
        for label in data.labels:
            res = str(tuple(label.shape[1:]))
            if res not in self._res_dict[data.split]:
                self._res_dict[data.split].update({res: 1})
            else:
                self._res_dict[data.split][res] += 1

    def _process(self):
        for split in ['train', 'val']:
            create_bar_plot(ax=self.ax, data=list(self._res_dict[split].values()), title='Labels resolutions',
                            labels=list(self._res_dict[split].keys()), y_label='# Of Labels', yticks=True,
                            split=split, ticks_rotation=0, color=self.colors[split], x_label='Resolution [W, H]')
            self.json_object.update({split: self._res_dict[split]})

