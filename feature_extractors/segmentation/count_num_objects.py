from typing import List

from utils.data_classes import BatchData
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class SegmentationCountNumObjects(SegmentationFeatureExtractorAbstract):
    def __init__(self, max_number_of_objects):
        super().__init__()
        self._thresh = max_number_of_objects
        self._hist: List[int] = [0] * (self._thresh + 1)
        self.labels = ['BG image', '1', '2', '3', '4', '5', '6', '7', '8', '9+']

    def execute(self, data: BatchData):
        for onehot_contours in data.batch_onehot_contours:
            num_objects_per_image = 0
            for cls_contours in onehot_contours:
                num_objects_per_image += len(cls_contours)
            self._hist[min(num_objects_per_image, self._thresh)] += 1

    def process(self, ax, train):
        # Normalize hist
        create_bar_plot(ax, self.normalize_hist(self._hist), self.labels, x_label="# Objects in image",
                        y_label="# Of images", title="# Objects per image", train=train, color=self.colors[int(train)])

        ax.grid(visible=True, axis='y')
