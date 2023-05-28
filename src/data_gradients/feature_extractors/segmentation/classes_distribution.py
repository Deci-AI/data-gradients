from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.utils import class_id_to_name
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults
from data_gradients.feature_extractors.utils import normalize_values_to_percentages


@register_feature_extractor()
class GetClassDistribution(FeatureExtractorAbstract):
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        ignore_labels = ignore_labels or []
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {"train": dict.fromkeys(keys, 0), "val": dict.fromkeys(keys, 0)}
        self._total_objects = {"train": 0, "val": 0}
        self.ignore_labels = ignore_labels

    def update(self, sample: SegmentationSample):
        for j, cls_contours in enumerate(sample.contours):
            if cls_contours:
                self._hist[sample.split][cls_contours[0].class_id] += len(cls_contours)
                self._total_objects[sample.split] += len(cls_contours)

    def _aggregate(self, split: str):
        self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
        values = normalize_values_to_percentages(self._hist[split].values(), self._total_objects[split])
        bins = self._hist[split].keys()

        results = HistogramResults(
            bin_names=bins,
            bin_values=values,
            plot="bar-plot",
            split=split,
            title="Classes distribution across dataset",
            color=self.colors[split],
            x_label="Class #",
            y_label="# Class instances [%]",
            y_ticks=True,
            ax_grid=True,
            values_to_log=list(self._hist[split].values()),
        )
        return results

    @property
    def description(self):
        return "Total # of connected components VS Class ID. \n" \
               "If the average number of components per image is too high, it might be due to image noise or the " \
               "presence of many small objects."
