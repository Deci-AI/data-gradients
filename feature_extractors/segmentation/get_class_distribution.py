import numpy as np

import preprocessing
from utils.data_classes import BatchData
from data_loaders.get_torch_loaders import sbd_label_to_class
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class SegmentationGetClassDistribution(SegmentationFeatureExtractorAbstract):
    def __init__(self, number_of_classes):
        super().__init__()
        self._hist = [0] * number_of_classes

    def execute(self, data: BatchData):
        for i, onehot_contours in enumerate(data.batch_onehot_contours):
            for cls_contours in onehot_contours:
                if len(cls_contours) == 0:
                    break
                else:
                    contours_class = preprocessing.contours.get_contour_class(cls_contours[0], data.labels[i])
                    # TODO: Check why sometimes contours class is 0 (debug mode)
                    # TODO: NOT WORKING
                    self._hist[min(contours_class - 1, len(self._hist)-1)] += len(cls_contours)

    def process(self, ax, train):

        d = {self._hist[i]: str(self._hist.index(self._hist[i])) for i, _ in enumerate(self._hist)}
        labels = [d[self._hist[i]] for i in range(len(self._hist))]

        self._hist = np.array(self._hist) / sum(self._hist)
        create_bar_plot(ax, self._hist, labels, x_label="Class", y_label="# Class instances",
                        title="Classes distribution", train=train, color=self.colors[int(train)])

        ax.grid(visible=True)

