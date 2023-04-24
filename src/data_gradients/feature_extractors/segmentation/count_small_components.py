from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HistogramResults


class CountSmallComponents(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    """

    def __init__(self, minimum_percent_of_an_image):
        super().__init__()
        self._min_size: float = minimum_percent_of_an_image / 100
        self._hist = {
            "train": {f"<{self._min_size}": 0},
            "val": {f"<{self._min_size}": 0},
        }
        self._total_objects = {"train": 0, "val": 0}

    def update(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            _, labels_h, labels_w = data.labels[i].shape
            self._total_objects[data.split] += sum([len(cls_contours) for cls_contours in image_contours])
            for class_contours in image_contours:
                for contour in class_contours:
                    self._hist[data.split][f"<{self._min_size}"] += 1 if contour.area < labels_w * labels_h * self._min_size else 0

    def _aggregate(self, split: str):
        values = self.normalize(self._hist[split].values(), self._total_objects[split])
        bins = list(self._hist[split].keys())

        results = HistogramResults(
            bins=bins,
            values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title=f"Components smaller then {self._min_size}% of image",
            x_label="% Components",
            y_label="",
            y_ticks=True,
            ax_grid=True,
        )
        return results
