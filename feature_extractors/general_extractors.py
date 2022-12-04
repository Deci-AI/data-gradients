from typing import List

import numpy as np
import torch
import torchvision

from batch_data import BatchData
from feature_extractors.feature_extractor_abstract import FeatureExtractorBuilder
from tensorboard_logger import create_bar_plot


class NumberOfImagesLabels(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        self._num_images: int = 0
        self._num_labels: int = 0

    def execute(self, data: BatchData):
        self._num_images += len(data.images)
        self._num_labels += len(data.labels)

    def process(self, ax):
        create_bar_plot(ax=ax, data=[self._num_images, self._num_labels], labels=["images", "labels"],
                        y_label='Total #', title='# Images & Labels', train=self.train_set,
                        ticks_rotation=0)


class NumberOfUniqueClasses(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        self._unique_classes = set()

    def execute(self, data):
        for label in data.labels:
            for val in label.unique():
                self._unique_classes.add(val.item())

    def process(self, ax):
        pass
        # print('Number of unique classes: ', len(self._unique_classes))


class ImagesResolutions(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        self._res_dict = dict()
        self._channels_last = False

    def execute(self, data):
        for image in data.images:
            res = tuple(image.shape[:-1] if self._channels_last else image.shape[1:])
            if res not in self._res_dict:
                self._res_dict[res] = 1
            else:
                self._res_dict[res] += 1

    def process(self, ax):
        pass
        # print('Resolutions dict: ', self._res_dict)


class ImagesAspectRatios(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        self._ar_dict = dict()
        self._channels_last = False

    def execute(self, data):
        for image in data.images:
            res = tuple(image.shape[:-1] if self._channels_last else image.shape[1:])
            ar = res[0] / res[1]
            if ar not in self._ar_dict:
                self._ar_dict[ar] = 1
            else:
                self._ar_dict[ar] += 1

    def process(self, ax):
        pass
        # print('Aspect ratio dict: ', self._ar_dict)


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


class AverageContrast(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
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

    def process(self, ax):
        # print('Contrast of images is: ', np.mean(self._contrast))
        pass


class NumberOfBackgroundImages(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        self._background_counter: int = 0

    def execute(self, data):
        for label in data.labels:
            self._background_counter += 1 if torch.sum(label).item() == 0 else 0

    def process(self, ax):
        pass
        # print('Number of background images is: ', self._background_counter)
