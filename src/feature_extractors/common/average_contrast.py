from typing import List

import numpy as np
import torch
import torchvision

from src.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class AverageContrast(FeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._grayscale: bool = True
        self._contrast: List[float] = []

    def execute(self, data):
        # if self._grayscale:
        #     images = self._get_gray_scaled_images()
        images = [torchvision.transforms.ToPILImage()(x) for x in data.images]
        images = [torchvision.transforms.Grayscale()(x) for x in images]
        images = [torchvision.transforms.ToTensor()(x) for x in images]
        images = torch.stack(images)
        self._contrast.append(np.round(torch.std(images).item(), 5))

    def process(self, ax, train):
        # print('Contrast of images is: ', np.mean(self._contrast))
        pass

