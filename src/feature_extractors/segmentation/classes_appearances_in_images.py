import numpy as np
import torch

from src.logger.logger_utils import create_bar_plot, create_json_object, class_id_to_name
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract


class AppearancesInImages(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    For each class, calculate percentage of images it appears in out of all images in set.
    """
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        self.ignore_labels = ignore_labels
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': dict.fromkeys(keys, 0), 'val': dict.fromkeys(keys, 0)}
        self._number_of_images = {'train': 0, 'val': 0}

    def _execute(self, data: SegBatchData):
        try:
            self._number_of_images[data.split] += len(data.labels)
            for label in data.labels:
                for u in label.unique():
                    u = int(u.item())
                    if u not in self.ignore_labels:
                        self._hist[data.split][u] += 1

        except Exception as e:
            print(self.__class__.__name__, e)

    def _process(self):
        for split in ['train', 'val']:
            # TODO: Split normalization / class id to name / create bar plot from _process
            self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
            values = self.normalize(self._hist[split].values(), self._number_of_images[split])
            create_bar_plot(self.ax, values, self._hist[split].keys(), x_label="Class #",
                            y_label="Images appeared in [%]", title="% Images that class appears in", split=split,
                            color=self.colors[split], yticks=True)
            self.ax.grid(visible=True)
            self.json_object.update({split: create_json_object(self._hist[split].values(), self._hist[split].keys())})
