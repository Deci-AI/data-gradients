from typing import List

import cv2
import numpy as np

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.logging.logger_utils import create_bar_plot, create_json_object
from src.utils import BatchData


class AverageBrightness(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._brightness = {'train': [], 'val': []}

    def _execute(self, data: BatchData):
        for image in data.images:
            np_image = image.numpy().transpose(1, 2, 0)
            lightness, _, _ = cv2.split(cv2.cvtColor(np_image, cv2.COLOR_BGR2LAB))
            # TODO: Handle zero division better
            if lightness is None:
                continue
            if np.max(lightness) == 0:
                continue
            n_lightness = lightness / np.max(lightness)
            self._brightness[data.split].append(np.mean(n_lightness))

    def _process(self):
        for split in ['train', 'val']:
            values, bins = self._post_process(self._brightness[split])
            create_bar_plot(self.ax, list(values), bins,
                            x_label="", y_label="% out of all images",
                            title="Average brightness of images",
                            split=split, color=self.colors[split], yticks=True)

            self.json_object.update({split: create_json_object(values, bins)})

    def _post_process(self, data, num_bins=10):
        values, bins = np.histogram(data, bins=num_bins)
        values = [np.round(((100 * value) / sum(list(values))), 3) for value in values]
        bins = self._create_keys(bins)
        return values, bins

    @staticmethod
    def _create_keys(bins):
        new_keys: List[str] = []
        for i, key in enumerate(bins):
            if i == 0:
                continue
            elif i == 1:
                new_keys.append('<%.2f' % key)
            elif i == len(bins) - 1:
                new_keys.append('%.2f<' % key)
            else:
                new_keys.append('%.2f<%.2f' % (key, bins[i+1]))
        return new_keys

