from typing import List
import numpy as np

import data_gradients.preprocess.contours
from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract


class SegmentationCountSmallComponents(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    TODO: NOT IMPLEMENTED WELL YET
    """
    def __init__(self, percent_of_an_image):
        super().__init__()
        # TODO NUMBERS DOES NOT WORK
        min_pixels: int = int(512 * 512 / (percent_of_an_image * 100))
        self.bins = np.array(range(0, min_pixels, int(min_pixels / 10)))
        self._hist: List[int] = [0] * 11
        self.label = ['<0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '>0.1']

    def _execute(self, data: SegBatchData):
        for i, onehot_contours in enumerate(data.batch_onehot_contours):
            for cls_contours in onehot_contours:
                for c in cls_contours:
                    _, _, contour_area = data_gradients.preprocess.contours.get_contour_moment(c)
                    self._hist[np.digitize(contour_area, self.bins) - 1] += 1

    def _post_process(self):
        # TODO: Make it work
        hist = list(np.array(self._hist) / sum(self._hist))
        create_bar_plot_old(ax, hist, self.label,
                            x_label="Object Size [%]", y_label="# Objects", ticks_rotation=0,
                            title="Number of small objects", split=split, color=self.colors[split])

        ax.grid(visible=True, axis='y')
        return dict(zip(self.label, hist))