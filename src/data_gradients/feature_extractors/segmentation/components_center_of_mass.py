import numpy as np

from data_gradients.utils.utils import class_id_to_name
from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.feature_extractor_abstract import (
    MultiFeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HeatMapResults


class ComponentsCenterOfMass(MultiFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all X, Y positions of center of mass of every object in every image for every class.
    Plot those X, Y positions as a heat-map
    """

    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {
            "train": {k: {"x": list(), "y": list()} for k in keys},
            "val": {k: {"x": list(), "y": list()} for k in keys},
        }

        self.num_axis = (1, 2)

    def update(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            label_shape = data.labels[0][0].shape
            for j, cls_contours in enumerate(image_contours):
                for contour in cls_contours:
                    self._hist[data.split][contour.class_id]["x"].append(round(contour.center[0] / label_shape[1], 2))
                    self._hist[data.split][contour.class_id]["y"].append(round(contour.center[1] / label_shape[0], 2))

    def aggregate_to_result(self, split: str):
        self._hist[split] = class_id_to_name(self.id_to_name, self._hist[split])
        x, y = self.aggregate(split)

        results = dict.fromkeys(self._hist[split])
        for key in self._hist[split]:
            n_bins = max(int(np.sqrt(len([key])) * 4), 20)
            results[key] = HeatMapResults(
                x=x[key],
                y=y[key],
                n_bins=n_bins,
                split=split,
                plot="heat-map",
                title="Center of mass average locations",
                x_label="X axis",
                y_label="Y axis",
                keys=["X", "Y"],
                range=[[0, 1], [0, 1]],
                invert=True,
            )

        # quantized_heat_map, _, _ = np.histogram2d(x, y, bins=25)
        # results.json_values = quantized_heat_map.tolist()

        return results

    def aggregate(self, split: str):
        # self._hist = self.merge_dict_splits(self._hist)
        x, y = {}, {}
        for key, val in self._hist[split].items():
            x[key] = val["x"]
            y[key] = val["y"]
        return x, y
