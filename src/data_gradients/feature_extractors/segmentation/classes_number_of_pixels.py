import numpy as np

from data_gradients.logging.logger_utils import class_id_to_name
from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from data_gradients.utils.data_classes.extractor_results import Results


class PixelsPerClass(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all sums of pixels per object per class and show the distribution among all classes
    """
    def __init__(self, num_classes, ignore_labels):
        super().__init__()

        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': {k: [] for k in keys}, 'val': {k: [] for k in keys}}
        self.ignore_labels = ignore_labels

    def _execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            img_dim = (data.labels[i].shape[1] * data.labels[i].shape[2])
            for j, cls_contours in enumerate(image_contours):
                for u in data.labels[i][j].unique():
                    u = int(u.item())
                    if u not in self.ignore_labels:
                        for contour in cls_contours:
                            size = np.round(100 * contour.area / img_dim, 3)
                            self._hist[data.split][u].append(size)

    def _post_process(self, split):
        values, bins = self._process_data(split)
        results = Results(bins=bins,
                          values=values,
                          plot='bar-plot',
                          split=split,
                          color=self.colors[split],
                          title="Average Pixels Per Component",
                          x_label="Class",
                          y_label="Size of component [% of image]",
                          y_ticks=True,
                          ax_grid=True
                          )
        return results

    def _process_data(self, split):
        self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
        hist = dict.fromkeys(self._hist[split].keys(), 0.)
        for cls in self._hist[split]:
            if len(self._hist[split][cls]):
                hist[cls] = float(np.round((np.mean(self._hist[split][cls])), 3))
        values = np.array(list(hist.values()))
        bins = self._hist[split].keys()
        return values, bins