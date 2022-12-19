import numpy as np

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class PixelsPerClass(SegmentationFeatureExtractorAbstract):
    def __init__(self, num_classes, ignore_labels):
        super().__init__()

        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {k: [] for k in keys}

    def execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            img_dim = (data.images[i].shape[1] * data.images[i].shape[2])
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                for contour in cls_contours:
                    self._hist[int(np.delete(unique, 0))].append(100 * contours.get_contour_area(contour) / img_dim)

    def process(self, ax, train):

        hist = dict.fromkeys(self._hist.keys(), 0.)
        for cls in self._hist:
            if len(self._hist[cls]):
                hist[cls] = float(np.mean(self._hist[cls]))
        hist_values = np.array(list(hist.values()))
        create_bar_plot(ax, hist_values, self._hist.keys(),
                        x_label="Class", y_label="Size of object [% of image]", title="Average Pixels Per Object",
                        train=train, color=self.colors[int(train)], yticks=True)

        ax.grid(visible=True, axis='y')
        return dict(zip(self._hist.keys(), hist_values))
