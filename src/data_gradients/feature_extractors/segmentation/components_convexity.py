import numpy as np

from data_gradients.logging.logger_utils import class_id_to_name
from data_gradients.preprocess import contours
from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from data_gradients.utils.data_classes.extractor_results import Results


class ComponentsConvexity(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': {k: [] for k in keys}, 'val': {k: [] for k in keys}}
        self.ignore_labels = ignore_labels

    def _execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                for u in data.labels[i][j].unique():
                    u = int(u.item())
                    if u not in self.ignore_labels:
                        for c in cls_contours:
                            convex_hull = contours.get_convex_hull(c)
                            convex_hull_perimeter = contours.get_contour_perimeter(convex_hull)
                            convexity_measure = (c.perimeter - convex_hull_perimeter) / c.perimeter
                            self._hist[data.split][u].append(convexity_measure)

    def _post_process(self, split):
        values, bins = self._process_data(split)
        results = Results(values=values,
                          bins=bins,
                          x_label="Class",
                          y_label="Convexity measure",
                          title="Convexity of components",
                          split=split,
                          color=self.colors[split],
                          y_ticks=True,
                          ax_grid=True,
                          plot='bar-plot')
        return results

    def _process_data(self, split: str):
        hist = dict.fromkeys(self._hist[split].keys(), 0.)
        for cls in self._hist[split]:
            if len(self._hist[split][cls]):
                hist[cls] = float(np.round(np.mean(self._hist[split][cls]), 3))
        hist = class_id_to_name(self.id_to_name, hist)
        values = np.array(list(hist.values()))
        bins = hist.keys()
        return values, bins