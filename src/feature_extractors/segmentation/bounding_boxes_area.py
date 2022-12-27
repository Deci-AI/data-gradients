import numpy as np

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot, create_json_object


class ComponentsSizeDistribution(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """
    def __init__(self, num_classes, ignore_labels):
        super().__init__()

        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': {k: [] for k in keys}, 'val': {k: [] for k in keys}}

    def _execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            img_dim = (data.labels[i].shape[1] * data.labels[i].shape[2])
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                for c in cls_contours:
                    rect = contours.get_rotated_bounding_rect(c)
                    wh = rect[1]
                    self._hist[data.split][int(np.delete(unique, 0))].append(100 * int(wh[0] * wh[1]) / img_dim)

    def _process(self):
        for split in ['train', 'val']:
            hist = dict.fromkeys(self._hist[split].keys(), 0.)
            for cls in self._hist[split]:
                if len(self._hist[split][cls]):
                    hist[cls] = float(np.round(np.mean(self._hist[split][cls]), 3))

            create_bar_plot(self.ax, list(hist.values()), hist.keys(), x_label="Class",
                            y_label="Size of BBOX [% of image]", title="Components Bounding-Boxes area",
                            split=split, color=self.colors[split], yticks=True)

            self.ax.grid(visible=True, axis='y')
            self.json_object.update({split: create_json_object(hist.values(), hist.keys())})



class ComponentsSizeDistributionOptimized(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """
    def __init__(self, num_classes, ignore_labels):
        super().__init__()

        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': {k: [] for k in keys}, 'val': {k: [] for k in keys}}

    def _execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            img_dim = (data.labels[i].shape[1] * data.labels[i].shape[2])
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                for c in cls_contours:
                    rect = contours.get_rotated_bounding_rect(c)
                    wh = rect[1]
                    self._hist[data.split][int(np.delete(unique, 0))].append(100 * int(wh[0] * wh[1]) / img_dim)

    def _process(self):
        for split in ['train', 'val']:
            hist = dict.fromkeys(self._hist[split].keys(), 0.)
            for cls in self._hist[split]:
                if len(self._hist[split][cls]):
                    hist[cls] = float(np.round(np.mean(self._hist[split][cls]), 3))

            create_bar_plot(self.ax, list(hist.values()), hist.keys(), x_label="Class",
                            y_label="Size of BBOX [% of image]", title="Components Bounding-Boxes area",
                            split=split, color=self.colors[split], yticks=True)

            self.ax.grid(visible=True, axis='y')
            self.json_object.update({split: create_json_object(hist.values(), hist.keys())})
