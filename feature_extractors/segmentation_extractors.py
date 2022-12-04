from typing import List
import numpy as np

import preprocessing.contours
from batch_data import BatchData
from create_torch_loaders import label_to_class
from feature_extractors.feature_extractor_abstract import FeatureExtractorBuilder
from tensorboard_logger import create_bar_plot


class SegmentationCountNumObjects(FeatureExtractorBuilder):
    def __init__(self, train_set, params):
        super().__init__(train_set)
        self._thresh = params['max_number_of_objects']
        self._hist: List[int] = [0] * self._thresh

    def execute(self, data: BatchData):
        for onehot_contours in data.batch_onehot_contours:
            num_objects_per_image = 0
            for cls_contours in onehot_contours:
                num_objects_per_image += len(cls_contours)
            self._hist[min(num_objects_per_image, self._thresh)] += 1

    def process(self, ax):
        # Normalize hist
        hist = list(np.array(self._hist) / sum(self._hist))
        # TODO: Fix hard coded labels number
        labels = ['BG image', '1', '2', '3', '4', '5', '6', '7', '8', '9+']
        create_bar_plot(ax, hist, labels, x_label="# Objects in image", y_label="# Of images",
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


class SegmentationCountSmallObjects(FeatureExtractorBuilder):
    def __init__(self, train_set, params):
        super().__init__(train_set)
        # TODO: Do params validation before running program
        assert 0 < params['percent_of_an_image'] < 100, "Param percent of image is a % of the image, in (0, 100)"
        min_pixels: int = int(512 * 512 / (params['percent_of_an_image'] * 100))
        self.bins = np.array(range(0, min_pixels, int(min_pixels / 10)))
        self._hist: List[int] = [0] * 11

    def execute(self, data: BatchData):
        for i, onehot_contours in enumerate(data.batch_onehot_contours):
            for cls_contours in onehot_contours:
                for c in cls_contours:
                    _, _, contour_area = preprocessing.contours.get_contour_moment(c)
                    self._hist[np.digitize(contour_area, self.bins) - 1] += 1

    def process(self, ax):
        # TODO: Fix hard-coded labels
        label = ['<0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '>0.1']

        self._hist = list(np.array(self._hist) / sum(self._hist))

        create_bar_plot(ax, self._hist, label,
                        x_label="Object Size [%]", y_label="# Objects", ticks_rotation=0,
                        title="Number of small objects", train=self.train_set)

        ax.grid(visible=True, axis='y')

        # ax.grid(visible=True)
