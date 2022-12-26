from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class ImagesResolutions(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._res_dict = {'train': dict(), 'val': dict()}

    def execute(self, data):
        for image in data.images:
            res = str(tuple(image.shape[1:]))
            if res not in self._res_dict[data.split]:
                self._res_dict[data.split][res] = 1
            else:
                self._res_dict[data.split][res] += 1

    def _process(self):
        for split in ['train', 'val']:
            create_bar_plot(ax=self.ax, data=list(self._res_dict[split].values()),
                            labels=list(self._res_dict[split].keys()), y_label='# Of Images',
                            title='Image resolutions', x_label='Resolution [W, H]', split=split, ticks_rotation=0,
                            color=self.colors[split], yticks=True)
            self.json_object.update({split: self._res_dict[split]})
        return self._res_dict


