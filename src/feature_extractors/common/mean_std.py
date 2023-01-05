import numpy as np
import torch

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logging.logger_utils import create_bar_plot, create_json_object
from src.utils import BatchData


class MeanAndSTD(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()

        self._hist = {'train': {'mean': [], 'std': []},
                      'val': {'mean': [], 'std': []}}

    def _execute(self, data: BatchData):
        for image in data.images:
            self._hist[data.split]['mean'].append(torch.mean(image, [1, 2]))
            self._hist[data.split]['std'].append(torch.std(image, [1, 2]))

    def _process(self):
        self.merge_dict_splits(self._hist)
        bgr_means = np.zeros(3)
        bgr_std = np.zeros(3)
        for split in ['train', 'val']:
            for channel in range(3):
                means = [self._hist[split]['mean'][i][channel].item() for i in range(len(self._hist[split]['mean']))]
                bgr_means[channel] = np.mean(means)
                stds = [self._hist[split]['std'][i][channel].item() for i in range(len(self._hist[split]['std']))]
                bgr_std[channel] = np.mean(stds)
            values = [bgr_means[0], bgr_std[0], bgr_means[1], bgr_std[1], bgr_means[2], bgr_std[2]]
            labels = ['Blue-Mean', 'Blue-STD', 'Green-Mean', 'Green-STD', 'Red-Mean', 'Red-STD']
            create_bar_plot(ax=self.ax, data=values, labels=labels, y_label='Mean / STD', title='Images mean & std',
                            split=split, ticks_rotation=0, color=self.colors[split], yticks=True)
            self.json_object.update({split: create_json_object(values, labels)})
