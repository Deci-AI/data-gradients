import numpy as np

from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_bar_plot, create_json_object, class_id_to_name


class GetClassDistribution(SegmentationFeatureExtractorAbstract):
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': dict.fromkeys(keys, 0), 'val': dict.fromkeys(keys, 0)}
        self._total_objects = {'train': 0, 'val': 0}
        self.ignore_labels = ignore_labels

    def _execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                for u in data.labels[i][j].unique():
                    u = int(u.item())
                    if u not in self.ignore_labels:
                        self._hist[data.split][u] += len(cls_contours)
                        self._total_objects[data.split] += len(cls_contours)

    def _process(self):
        for split in ['train', 'val']:
            self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
            values = self.normalize(self._hist[split].values(), self._total_objects[split])
            create_bar_plot(self.ax, values, self._hist[split].keys(), x_label="Class #",
                            y_label="# Class instances [%]", title="Classes distribution across dataset",
                            split=split, color=self.colors[split], yticks=True)

            self.ax.grid(visible=True)
            self.json_object.update({split: create_json_object(self._hist[split].values(), self._hist[split].keys())})
