import numpy as np

from utils.data_classes import BatchData
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class GetClassDistribution(SegmentationFeatureExtractorAbstract):
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = dict.fromkeys(keys, 0)

    def execute(self, data: BatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                self._hist[int(np.delete(unique, 0))] += len(cls_contours)

    def process(self, ax, train):

        create_bar_plot(ax, self._hist.values(), self._hist.keys(), x_label="Class", y_label="# Class instances",
                        title="Classes distribution", train=train, color=self.colors[int(train)])

        ax.grid(visible=True)

