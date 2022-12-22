import numpy as np

from src.logger.logger_utils import create_bar_plot
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract


class AppearancesInImages(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    For each class, calculate percentage of images it appears in out of all images in set.
    """
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = dict.fromkeys(keys, 0)
        self._number_of_images: int = 0

    def execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            self._number_of_images += 1
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                self._hist[int(np.delete(unique, 0))] += 1

    def process(self, ax, train):
        values = self.normalize(self._hist.values(), self._number_of_images)
        create_bar_plot(ax, values, self._hist.keys(), x_label="Class #", y_label="Images appeared in [%]",
                        title="% Images that class appears in", train=train, color=self.colors[int(train)], yticks=True)
        ax.grid(visible=True)

        return dict(zip(self._hist.keys(), self._hist.values()))
