from typing import List

import numpy as np

from logger.logger_utils import create_heatmap_plot
from preprocessing import contours
from utils.data_classes import BatchData
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract


class ObjectsCenterOfMass(SegmentationFeatureExtractorAbstract):
    ERROR_CENTER = (-1, -1)

    def __init__(self):
        super().__init__()
        self._x: List = []
        self._y: List = []
        self._sigma: float = 0
        self._bins: int = 0
        self.single_axis = False

    def execute(self, data: BatchData):
        for i, image_contours in enumerate(data.contours):
            for cls_contours in image_contours:
                for c in cls_contours:
                    center = contours.get_contour_center_of_mass(c)
                    if center != self.ERROR_CENTER:
                        self._x.append(center[0])
                        self._y.append(center[1])

    def process(self, ax, train):
        # TODO: My thumb rules numbers
        self._bins = int(np.sqrt(len(self._x)) * 4)
        self._sigma = 2 * (self._bins / 150)

        create_heatmap_plot(ax=ax, x=self._x, y=self._y,  train=train, bins=self._bins, sigma=self._sigma,
                            title=f'Center of mass average locations', x_label='X axis', y_label='Y axis')
