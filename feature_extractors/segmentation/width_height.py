from typing import List
import numpy as np

import preprocess.contours
from preprocess import contours
from utils.data_classes import BatchData
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot, create_heatmap_plot


class WidthHeight(SegmentationFeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._width: List = []
        self._height: List = []
        self.single_axis = False

    def execute(self, data: BatchData):
        for i, image_contours in enumerate(data.contours):
            for cls_contours in image_contours:
                for c in cls_contours:
                    extreme_points = contours.get_extreme_points(c)
                    self._width.append(extreme_points["rightmost"][0] - extreme_points["leftmost"][0])
                    self._height.append(extreme_points["bottommost"][1] - extreme_points["topmost"][1])

    def process(self, ax, train):

        width = []
        height = []
        for w, h in zip(self._width, self._height):
            if w == 0 or h == 0:
                continue
            else:
                width.append(w)
                height.append(h)
        create_heatmap_plot(ax=ax, x=width, y=height, train=train, bins=25,
                            sigma=0, title=f'Width / Height', x_label='Width', y_label='Height')
