import numpy as np

from preprocess import contours
from utils.data_classes import BatchData
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class ObjectSizeDistribution(SegmentationFeatureExtractorAbstract):
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
                for c in cls_contours:
                    box = contours.get_rotated_bounding_rect(c)
                    wh = box[1]
                    self._hist[int(np.delete(unique, 0))].append(int(wh[0] * wh[1]))
                    # if wh[0] * wh[1] < 1:
                    #     print(c)
                    #     print(f'Appended {int(wh[0] * wh[1])}')

    def process(self, ax, train):
        hist = dict.fromkeys(self._hist.keys(), 0.)
        for cls in self._hist:
            if len(self._hist[cls]):
                hist[cls] = float(np.mean(self._hist[cls]))
        create_bar_plot(ax, hist.values(), self._hist.keys(),
                        x_label="Class", y_label="Size of BBOX", title="Objects BBOX size",
                        train=train, color=self.colors[int(train)])

        ax.grid(visible=True, axis='y')
