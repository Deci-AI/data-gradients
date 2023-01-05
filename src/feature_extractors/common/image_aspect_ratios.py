import numpy as np

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logging.logger_utils import create_bar_plot
from src.utils import BatchData


class ImagesAspectRatios(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._hist = {'train': dict(), 'val': dict()}

    def _execute(self, data: BatchData):
        for image in data.images:
            ar = np.round(image.shape[2] / image.shape[1], 2)
            if ar not in self._hist[data.split]:
                self._hist[data.split][ar] = 1
            else:
                self._hist[data.split][ar] += 1

    def _process(self):
        self.merge_dict_splits(self._hist)

        for split in ['train', 'val']:
            create_bar_plot(ax=self.ax, data=list(self._hist[split].values()),
                            labels=list(self._hist[split].keys()), y_label='# Of Images',
                            title='Image aspect ratios', x_label='Aspect ratio [W / H]', split=split, ticks_rotation=0,
                            color=self.colors[split], yticks=True)
            self.json_object.update({split: self._hist[split]})
