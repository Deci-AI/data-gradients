import numpy as np

from data_gradients.logging.logger_utils import class_id_to_name
from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import Results


class ComponentsSizeDistribution(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()

        keys = [
            int(i)
            for i in range(0, num_classes + len(ignore_labels))
            if i not in ignore_labels
        ]
        self._hist = {"train": {k: [] for k in keys}, "val": {k: [] for k in keys}}
        self.ignore_labels = ignore_labels

    def _execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            img_dim = data.labels[i].shape[1] * data.labels[i].shape[2]
            for class_channel in image_contours:
                for contour in class_channel:
                    self._hist[data.split][contour.class_id].append(
                        100 * int(contour.bbox_area) / img_dim
                    )

    def _post_process(self, split: str):
        values, bins = self._process_data(split)
        results = Results(
            bins=bins,
            values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title="Components Bounding-Boxes area",
            x_label="Class",
            y_label="Size of BBOX [% of image]",
            ax_grid=True,
            y_ticks=True,
        )
        return results

    def _process_data(self, split: str):
        self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
        hist = dict.fromkeys(self._hist[split].keys(), 0.0)
        for cls in self._hist[split]:
            if len(self._hist[split][cls]):
                hist[cls] = float(np.round(np.mean(self._hist[split][cls]), 3))
        values = list(hist.values())
        bins = hist.keys()
        return values, bins
