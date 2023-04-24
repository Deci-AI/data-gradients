from data_gradients.logging.logger_utils import class_id_to_name
from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults


class GetClassDistribution(FeatureExtractorAbstract):
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {"train": dict.fromkeys(keys, 0), "val": dict.fromkeys(keys, 0)}
        self._total_objects = {"train": 0, "val": 0}
        self.ignore_labels = ignore_labels

    def update(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                if cls_contours:
                    self._hist[data.split][cls_contours[0].class_id] += len(cls_contours)
                    self._total_objects[data.split] += len(cls_contours)

    def _aggregate_to_result(self, split: str):
        values, bins = self._aggregate(split)
        results = HistogramResults(
            bins=bins,
            values=values,
            plot="bar-plot",
            split=split,
            title="Classes distribution across dataset",
            color=self.colors[split],
            x_label="Class #",
            y_label="# Class instances [%]",
            y_ticks=True,
            ax_grid=True,
            json_values=self._hist[split].values(),
        )
        return results

    def _aggregate(self, split: str):
        self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
        values = self.normalize(self._hist[split].values(), self._total_objects[split])
        bins = self._hist[split].keys()
        return values, bins
