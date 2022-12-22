from typing import List

import numpy as np

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_heatmap_plot


class WidthHeight(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all Width, Height of bounding-box for every object in every image for every class.
    Plot those W, H values as a heat-map
    """
    def __init__(self):
        super().__init__()
        self._width: List = []
        self._height: List = []
        self.single_axis = False

    def execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for cls_contours in image_contours:
                for c in cls_contours:
                    rect = contours.get_rotated_bounding_rect(c)
                    self._width.append(rect[1][0])
                    self._height.append(rect[1][1])

    def process(self, ax, train):
        width = [w for w in self._width if w > 0]
        height = [h for h in self._height if h > 0]
        create_heatmap_plot(ax=ax, x=width, y=height, train=train, bins=10,
                            sigma=2, title=f'Bounding Boxes Width / Height', x_label='Width [px]', y_label='Height [px]',
                            use_gaussian_filter=True, use_extent=True)

        quantized_heat_map, _, _ = np.histogram2d(width, height, bins=25)
        return {"Quantized width height values": quantized_heat_map.tolist()}
