import numpy as np

from src.logger.logger_utils import create_bar_plot, create_json_object
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract


class AppearancesInImages(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    For each class, calculate percentage of images it appears in out of all images in set.
    """
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': dict.fromkeys(keys, 0), 'val': dict.fromkeys(keys, 0)}
        self._number_of_images = {'train': 0, 'val': 0}

    def execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            self._number_of_images[data.split] += 1
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                self._hist[data.split][int(np.delete(unique, 0))] += 1

    def _process(self):
        for split in ['train', 'val']:
            values = self.normalize(self._hist[split].values(), self._number_of_images[split])
            create_bar_plot(self.ax, values, self._hist[split].keys(), x_label="Class #",
                            y_label="Images appeared in [%]", title="% Images that class appears in", split=split,
                            color=self.colors[split], yticks=True)
            self.ax.grid(visible=True)
            self.json_object.update({split: create_json_object(self._hist[split].values(), self._hist[split].keys())})
