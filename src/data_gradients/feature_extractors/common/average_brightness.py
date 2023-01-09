from typing import List

import cv2
import numpy as np

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from src.utils import BatchData
from src.utils.data_classes import Results


class AverageBrightness(FeatureExtractorAbstract):
    # TODO: Check correlation between mean & std of an image and its brightness.
    #  Currently, ones == 1, zeros == 0, but N(0.5, 1)?
    def __init__(self):
        super().__init__()
        self._num_bins: int = 10
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
            print(np.mean(n_lightness))

    def _post_process(self, split: str) -> Results:
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

    def _process_data(self, split):
        values, bins = np.histogram(self._brightness[split], bins=self._num_bins)
        values = [np.round(((100 * value) / sum(list(values))), 3) for value in values]
        bins = self._create_keys(bins)
        return values, bins

    @staticmethod
    def _create_keys(bins):
        new_keys: List[str] = []
        for i, key in enumerate(bins):
            if i == len(bins) - 1:
                continue
            new_keys.append('{:.2f}<{:.2f}'.format(bins[i], bins[i + 1]))

        return new_keys

