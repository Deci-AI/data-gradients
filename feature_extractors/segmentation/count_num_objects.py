from typing import List

import numpy as np

from batch_data import BatchData
from feature_extractors import FeatureExtractorBuilder
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
