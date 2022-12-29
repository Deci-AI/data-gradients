import numpy as np

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot, create_json_object, class_id_to_name


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
                            size = np.round(100 * contours.get_contour_area(contour) / img_dim, 3)
                            self._hist[data.split][u].append(size)

    def _process(self):
        for split in ['train', 'val']:
            self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
            hist = dict.fromkeys(self._hist[split].keys(), 0.)
            for cls in self._hist[split]:
                if len(self._hist[split][cls]):
                    hist[cls] = float(np.round((np.mean(self._hist[split][cls])), 3))
            hist_values = np.array(list(hist.values()))
            create_bar_plot(self.ax, hist_values, self._hist[split].keys(),
                            x_label="Class", y_label="Size of component [% of image]", title="Average Pixels Per Component",
                            split=split, color=self.colors[split], yticks=True)

            self.ax.grid(visible=True, axis='y')
            self.json_object.update({split: create_json_object(hist_values, self._hist[split].keys())})
