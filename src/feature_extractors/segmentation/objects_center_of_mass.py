from typing import List

import numpy as np

from src.logger.logger_utils import create_heatmap_plot
from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract


class ObjectsCenterOfMass(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all X, Y positions of center of mass of every object in every image for every class.
    Plot those X, Y positions as a heat-map
    """
    def __init__(self):
        super().__init__()
        self._x: List = []
        self._y: List = []
        self._sigma: float = 0
        self._bins: int = 50
        self.single_axis = False

    def execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for cls_contours in image_contours:
                for c in cls_contours:
                    center = contours.get_contour_center_of_mass(c)
                    self._x.append(center[0])
                    self._y.append(center[1])

    def process(self, ax, train):
        # TODO: My thumb rules numbers
        self._bins = int(np.sqrt(len(self._x)) * 4)
        self._sigma = 2 * (self._bins / 150)

        # TODO: Divide each plot for a class. Need to make x, y as a dictionaries (every class..)

        create_heatmap_plot(ax=ax, x=self._x, y=self._y,  train=train, bins=self._bins, sigma=self._sigma,
                            title=f'Center of mass average locations', x_label='X axis', y_label='Y axis')

        return {"Am I implemented?": False}