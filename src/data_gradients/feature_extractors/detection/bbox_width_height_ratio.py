import numpy as np

from data_gradients.utils import DetectionBatchData
from data_gradients.utils.data_classes.extractor_results import HeatMapResults
from data_gradients.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract


class BBoxWidthHeightRatio(FeatureExtractorAbstract):
    """Compute the ratio of the bounding boxes width and height.
    Plot those W, H values as a heat-map
    """

    def __init__(self):
        super().__init__()
        self._width = {"train": [], "val": []}
        self._height = {"train": [], "val": []}
        self.num_axis = (1, 2)

    def update(self, data: DetectionBatchData):
        for i, (image, bboxes) in enumerate(zip(data.images, data.bboxes)):
            height, width = image.shape[1:]
            for j, bbox in enumerate(bboxes):
                bbox_height = bbox[3] - bbox[1]
                bbox_width = bbox[2] - bbox[0]
                self._width[data.split].append(bbox_width / width)
                self._height[data.split].append(bbox_height / height)

    def _aggregate(self, split: str):
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
        results.values_to_log = quantized_heat_map.tolist()
        results.keys = ["Width", "Height"]
        return results
