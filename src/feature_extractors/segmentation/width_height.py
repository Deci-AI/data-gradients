from typing import List

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_heatmap_plot


class WidthHeight(SegmentationFeatureExtractorAbstract):
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
        create_heatmap_plot(ax=ax, x=width, y=height, train=train, bins=50,
                            sigma=0, title=f'Width / Height', x_label='Width [px]', y_label='Height [px]')
        return {'Am I implemented?': False}
