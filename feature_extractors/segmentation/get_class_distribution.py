import numpy as np

import preprocessing
from batch_data import BatchData
from data_loaders.get_torch_loaders import sbd_label_to_class
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from tensorboard_logger import create_bar_plot


class SegmentationGetClassDistribution(SegmentationFeatureExtractorAbstract):
    def __init__(self, train_set, number_of_classes):
        super().__init__(train_set)
        self._hist = [0] * number_of_classes

    def execute(self, data: BatchData):
        for i, onehot_contours in enumerate(data.batch_onehot_contours):
            for cls_contours in onehot_contours:
                contours_class = preprocessing.contours.get_contour_class(cls_contours[0], data.labels[i])
                self._hist[contours_class - 1] += len(cls_contours)

    def process(self, ax):
        # TODO: Remove hard coded label-to-class
        d = {self._hist[i]: sbd_label_to_class[self._hist.index(self._hist[i])] for i, _ in enumerate(self._hist)}
        labels = [d[self._hist[i]] for i in range(len(self._hist))]

        self._hist = np.array(self._hist) / sum(self._hist)
        # TODO: Colors into base-class-dict
        create_bar_plot(ax, self._hist, labels, x_label="Class", y_label="# Class instances",
                        title="Classes distribution", train=self.train_set)

        ax.grid(visible=True)

