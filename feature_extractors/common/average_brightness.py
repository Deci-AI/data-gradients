from typing import List

import numpy as np
import torch

from feature_extractors import FeatureExtractorBuilder


class AverageBrightness(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        self._brightness: List[float] = []
        self._luminance: List[float] = []

    def execute(self, data):
        for image in data.images:
            self._brightness.append(np.round(torch.sum(image) / 3 / (image.shape[1] * image.shape[2]), 5))
            # self._luminance.append((0.2126 * image[0]) + (0.7152 * image[1]) + (0.0722 * image[2]))

    def process(self, ax):
        pass
        # print('Average brightness: {}'.format(np.mean(self._brightness)))  #, np.mean(self._luminance)))

