from typing import List
import numpy as np

import preprocessing.contours
from batch_data import BatchData
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class SegmentationCountSmallObjects(SegmentationFeatureExtractorAbstract):
    def __init__(self, train_set, params):
        super().__init__(train_set)
        # TODO: Do params validation before running program
        assert 0 < params['percent_of_an_image'] < 100, "Param percent of image is a % of the image, in (0, 100)"

        min_pixels: int = int(512 * 512 / (params['percent_of_an_image'] * 100))
        self.bins = np.array(range(0, min_pixels, int(min_pixels / 10)))
        # TODO: Magic number
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
                        title="Number of small objects", train=self.train_set, color=self.colors[int(self.train_set)])

        ax.grid(visible=True, axis='y')
