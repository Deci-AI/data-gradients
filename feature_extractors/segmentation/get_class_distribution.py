import numpy as np
from collections import defaultdict
import preprocessing
from utils.data_classes import BatchData
from data_loaders.get_torch_loaders import sbd_label_to_class
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class SegmentationGetClassDistribution(SegmentationFeatureExtractorAbstract):
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = dict.fromkeys(keys, 0)

    def execute(self, data: BatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                cls = int(np.delete(np.unique(data.labels[i][j]), 0))
                self._hist[cls] += len(cls_contours)

    def process(self, ax, train):

        create_bar_plot(ax, self._hist.values(), self._hist.keys(), x_label="Class", y_label="# Class instances",
                        title="Classes distribution", train=train, color=self.colors[int(train)])

        ax.grid(visible=True)

