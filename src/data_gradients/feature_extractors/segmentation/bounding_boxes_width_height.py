import numpy as np

from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.extractor_results import HeatMapResults


class WidthHeight(FeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all Width, Height of bounding-box for every object in every image for every class.
    Plot those W, H values as a heat-map
    """

    def __init__(self):
        super().__init__()
        self._width = {"train": [], "val": []}
        self._height = {"train": [], "val": []}
        self.num_axis = (1, 2)

    def update(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for j, class_channel in enumerate(image_contours):
                height, width = data.labels[i][j].shape
                for contour in class_channel:
                    self._width[data.split].append(contour.w / width)
                    self._height[data.split].append(contour.h / height)

    def _aggregate_to_result(self, split: str):
        width = [w for w in self._width[split] if w > 0]
        height = [h for h in self._height[split] if h > 0]

        results = HeatMapResults(
            x=width,
            y=height,
            n_bins=16,
            split=split,
            plot="heat-map",
            title="Bounding Boxes Width / Height",
            x_label="Width [% of image]",
            y_label="Height [% of image]",
            keys=["Width", "Height"],
        )

        quantized_heat_map, _, _ = np.histogram2d(width, height, bins=25)
        results.json_values = quantized_heat_map.tolist()
        results.keys = ["Width", "Height"]
        return results
