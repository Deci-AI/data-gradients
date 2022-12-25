import cv2
import numpy as np
import torch

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class ErosionTest(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {k: 0 for k in keys}
        self._hist_eroded = {k: 0 for k in keys}
        self._kernel = np.ones((3, 3), np.uint8)

    def execute(self, data: SegBatchData):

        for i, image_contours in enumerate(data.contours):
            label = data.labels[i].numpy().transpose(1, 2, 0).astype(np.uint8)
            eroded_label = cv2.erode(label, self._kernel, iterations=3)
            eroded_contours = contours.get_contours(torch.tensor(eroded_label).permute(2, 0, 1))
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                self._hist[int(np.delete(unique, 0))] += len(cls_contours)
                if eroded_contours:
                    self._hist_eroded[int(np.delete(unique, 0))] += len(eroded_contours)

                print(f'Adding {len(cls_contours)} to the regular hist, and adding {len(eroded_contours)} to the eroded list!')
                if len(eroded_contours ) > len(cls_contours):
                    print('WTF')

    def process(self, ax, train):
        hist = dict.fromkeys(self._hist.keys(), 0.)
        for cls in self._hist:
            if (self._hist[cls]) > 0:
                hist[cls] = np.round(100 * (self._hist_eroded[cls] / self._hist[cls]), 3)
            else:
                hist[cls] = 0

        hist_values = np.array(list(hist.values()))
        create_bar_plot(ax, hist_values, self._hist.keys(), x_label="Class",
                        y_label="% of disappearing contours after Erosion", title="Erosion & contours comparing",
                        train=train, color=self.colors[int(train)], yticks=True)

        ax.grid(visible=True, axis='y')
        return dict(zip(self._hist.keys(), hist_values))
