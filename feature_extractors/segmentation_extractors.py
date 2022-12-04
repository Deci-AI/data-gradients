from typing import List
import numpy as np

import preprocessing.contours
from batch_data import BatchData
from create_torch_loaders import label_to_class
from feature_extractors.feature_extractor_abstract import FeatureExtractorBuilder
from tensorboard_logger import create_bar_plot


class SegmentationCountNumObjects(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        # 51 random number
        self._hist: List[int] = [0] * 51

    def execute(self, data: BatchData):
        for onehot_contours in data.batch_onehot_contours:
            num_objects_per_image = 0
            for cls_contours in onehot_contours:
                num_objects_per_image += len(cls_contours)
            self._hist[num_objects_per_image] += 1

    def process(self, ax):
        # Cut hist from 51 (random number) to the highest # of objects found in data set
        idx = len(self._hist)
        for i, val in enumerate(reversed(self._hist)):
            if self._hist[-i] > 0:
                idx = len(self._hist) - i + 1
                break

        hist = self._hist[:idx]
        # Normalize hist
        hist = list(np.array(hist) / sum(hist))

        create_bar_plot(ax, hist, range(len(hist)), x_label="# Objects in image", y_label="# Of images",
                        title="# Objects per image", train=self.train_set)

        ax.grid(visible=True, axis='y')


class SegmentationGetClassDistribution(FeatureExtractorBuilder):
    def __init__(self, train_set, params):
        super().__init__(train_set)
        self._hist = [0] * params['number_of_classes']

    def execute(self, data: BatchData):
        for i, onehot_contours in enumerate(data.batch_onehot_contours):
            for cls_contours in onehot_contours:
                contours_class = preprocessing.contours.get_contour_class(cls_contours[0], data.labels[i])
                self._hist[contours_class - 1] += len(cls_contours)

    def process(self, ax):
        d = {self._hist[i]: label_to_class[self._hist.index(self._hist[i])] for i, _ in enumerate(self._hist)}
        labels = [d[self._hist[i]] for i in range(len(self._hist))]

        self._hist = np.array(self._hist) / sum(self._hist)

        create_bar_plot(ax, self._hist, labels, x_label="Class", y_label="# Class instances",
                        title="Classes distribution", train=self.train_set)

        ax.grid(visible=True)

