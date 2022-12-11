import numpy as np

from preprocessing import contours
from utils.data_classes import BatchData
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class PixelsPerClass(SegmentationFeatureExtractorAbstract):
    def __init__(self, num_classes, ignore_labels):
        super().__init__()

        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {k: [] for k in keys}

    def execute(self, data: BatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                for contour in cls_contours:
                    self._hist[int(np.delete(unique, 0))].append(contours.get_contour_area(contour))

    def process(self, ax, train):
        hist = dict.fromkeys(self._hist.keys(), 0.)
        for cls in self._hist:
            if len(self._hist[cls]):
                hist[cls] = np.mean(self._hist[cls])

        create_bar_plot(ax, hist.values(), self._hist.keys(),
                        x_label="Class", y_label="Average # Pixels per object", title="Average Pixels Per Object",
                        train=train, color=self.colors[int(train)])

        ax.grid(visible=True, axis='y')
