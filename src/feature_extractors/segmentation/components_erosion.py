import cv2
import numpy as np
import torch

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot, create_json_object, class_id_to_name


class ErosionTest(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': {k: 0. for k in keys}, 'val': {k: 0. for k in keys}}
        self._hist_eroded = {'train': {k: 0. for k in keys}, 'val': {k: 0. for k in keys}}
        self._kernel = np.ones((3, 3), np.uint8)
        self.ignore_labels = ignore_labels

    def execute(self, data: SegBatchData):

        for i, image_contours in enumerate(data.contours):
            label = data.labels[i].numpy().transpose(1, 2, 0).astype(np.uint8)
            eroded_label = cv2.erode(label, self._kernel, iterations=2)
            eroded_label_tensor = torch.tensor(eroded_label)
            if len(eroded_label_tensor.shape) == 2:
                eroded_label_tensor = eroded_label_tensor.unsqueeze(-1)
            eroded_contours = contours.get_contours(eroded_label_tensor.permute(2, 0, 1))
            for j, cls_contours in enumerate(image_contours):
                for u in data.labels[i][j].unique():
                    u = int(u.item())
                    if u not in self.ignore_labels:
                        self._hist[data.split][u] += len(cls_contours)
                        if eroded_contours:
                            self._hist_eroded[data.split][u] += len(eroded_contours)
    def _process(self):
        for split in ['train', 'val']:
            hist = dict.fromkeys(self._hist[split].keys(), 0.)
            for cls in self._hist[split]:
                if (self._hist[split][cls]) > 0:
                    hist[cls] = np.round(100 * (self._hist_eroded[split][cls] / self._hist[split][cls]), 3)
                else:
                    hist[cls] = 0

            hist = class_id_to_name(self.id_to_name, hist)
            hist_values = np.array(list(hist.values()))
            create_bar_plot(self.ax, hist_values, hist.keys(), x_label="Class",
                            y_label="% of disappearing contours after Erosion", title="Erosion & contours comparing",
                            split=split, color=self.colors[split], yticks=True)

            self.ax.grid(visible=True, axis='y')
            self.json_object.update({split: create_json_object(hist_values, self._hist[split].keys())})
