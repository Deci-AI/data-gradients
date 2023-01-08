from typing import List

import cv2
import numpy as np

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.utils import BatchData
from src.utils.data_classes import Results


class AverageBrightness(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._brightness = {'train': [], 'val': []}

    def _execute(self, data: BatchData):
        for image in data.images:
            np_image = image.numpy().transpose(1, 2, 0)
            lightness, _, _ = cv2.split(cv2.cvtColor(np_image, cv2.COLOR_BGR2LAB))
            if lightness is None:
                continue
            if np.all(lightness == 0) or np.max(lightness) == 0:
                n_lightness = 0
            else:
                n_lightness = lightness / np.max(lightness)
            self._brightness[data.split].append(np.mean(n_lightness))

    def _post_process(self, split: str):
        values, bins = self._process_data(split)
        results = Results(bins=bins,
                          values=list(values),
                          plot='bar-plot',
                          split=split,
                          title="Average brightness of images",
                          color=self.colors[split],
                          y_label="% out of all images",
                          y_ticks=True)
        return results

    def _process_data(self, split, num_bins=10):
        values, bins = np.histogram(self._brightness[split], bins=num_bins)
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

