import numpy as np

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot


class ComponentsConvexity(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {k: [] for k in keys}

    def execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                for c in cls_contours:
                    contour_area = contours.get_contour_area(c)
                    convex_hull_area = contours.get_contour_area(contours.get_convex_hull(c))
                    convexity_measure = (convex_hull_area - contour_area) / convex_hull_area
                    self._hist[int(np.delete(unique, 0))].append(convexity_measure)

    def process(self, ax, train):
        hist = dict.fromkeys(self._hist.keys(), 0.)
        for cls in self._hist:
            if len(self._hist[cls]):
                hist[cls] = float(np.mean(self._hist[cls]))
        hist_values = np.array(list(hist.values()))
        create_bar_plot(ax, hist_values, self._hist.keys(),
                        x_label="Class", y_label="Convexity measure", title="Convexity of objects",
                        train=train, color=self.colors[int(train)], yticks=True)

        ax.grid(visible=True, axis='y')
        return dict(zip(self._hist.keys(), hist_values))
