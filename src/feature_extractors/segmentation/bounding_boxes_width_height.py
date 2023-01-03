import numpy as np

from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.logger.logger_utils import create_heatmap_plot, create_json_object


class WidthHeight(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all Width, Height of bounding-box for every object in every image for every class.
    Plot those W, H values as a heat-map
    """
    def __init__(self):
        super().__init__()
        self._width = {'train': [], 'val': []}
        self._height = {'train': [], 'val': []}
        self.num_axis = (1, 2)

    def _execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                height, width = data.labels[i][j].shape
                for c in cls_contours:
                    # TODO: Add more logic to that, somehow
                    self._width[data.split].append(c.w / width)
                    self._height[data.split].append(c.h / height)

    def _process(self):
        for split in ['train', 'val']:
            width = [w for w in self._width[split] if w > 0]
            height = [h for h in self._height[split] if h > 0]
            create_heatmap_plot(ax=self.ax[int(split != 'train')], x=width, y=height, split=split, bins=16, sigma=8,
                                title=f'Bounding Boxes Width / Height', x_label='Width [% of image]',
                                y_label='Height [% of image]')

            quantized_heat_map, _, _ = np.histogram2d(width, height, bins=25)
            self.json_object.update({split: create_json_object(quantized_heat_map.tolist(), ["Width", "Height"])})
