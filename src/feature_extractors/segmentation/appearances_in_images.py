import numpy as np

from src.logger.logger_utils import create_bar_plot
from src.utils import BatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract


class AppearancesInImages(SegmentationFeatureExtractorAbstract):

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = dict.fromkeys(keys, 0)
        self._number_of_images: int = 0

    def execute(self, data: BatchData):
        for i, image_contours in enumerate(data.contours):
            self._number_of_images += 1
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                self._hist[int(np.delete(unique, 0))] += 1

    def process(self, ax, train):
        values = [((100 * value) / self._number_of_images) for value in self._hist.values()]
        create_bar_plot(ax, values, self._hist.keys(), x_label="Class", y_label="% Images appeared in",
                        title="% Appearances in images", train=train, color=self.colors[int(train)])

        ax.grid(visible=True)
