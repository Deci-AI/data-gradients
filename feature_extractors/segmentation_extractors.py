from typing import List
import numpy as np

from feature_extractors.feature_extractor_abstract import FeatureExtractorBuilder
from tensorboard_logger import create_bar_plot


class SegmentationCountNumObjects(FeatureExtractorBuilder):
    def __init__(self, train_set):
        super().__init__(train_set)
        # 51 random number
        self._hist: List[int] = [0] * 51

    def execute(self, data):
        for onehot_contours in data.batch_onehot_contours:
            num_objects_per_image = 0
            for cls_contours in onehot_contours:
                num_objects_per_image += len(cls_contours)
            self._hist[num_objects_per_image] += 1

    def process(self, ax):
        # Cut hist from 51 (random number) to the highest # of objects found in data set
        idx = len(self._hist)
        for i, val in enumerate(reversed(self._hist)):
            if self._hist[-i] > 0:
                idx = len(self._hist) - i + 1
                break

        hist = self._hist[:idx]
        # Normalize hist
        hist = list(np.array(hist) / sum(hist))

        create_bar_plot(ax, hist, range(len(hist)), x_label="# Objects in image", y_label="# Of images",
                        title="# Objects per image", train=self.train_set)

        ax.grid(visible=True, axis='y')

