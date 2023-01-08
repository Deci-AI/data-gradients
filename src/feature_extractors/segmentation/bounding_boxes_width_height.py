import numpy as np

from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.utils.data_classes.extractor_results import HeatMapResults


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

    def _post_process(self, split):
        x, y = self._process_data(split)
        results = HeatMapResults(x=x,
                                 y=y,
                                 n_bins=16,
                                 sigma=8,
                                 split=split,
                                 plot='heat-map',
                                 title=f'Bounding Boxes Width / Height',
                                 x_label='Width [% of image]',
                                 y_label='Height [% of image]'
                                 )

        quantized_heat_map, _, _ = np.histogram2d(x, y, bins=25)
        results.json_values = quantized_heat_map.tolist()
        results.keys = ["Width", "Height"]
        return results

    def _process_data(self, split: str):
        width = [w for w in self._width[split] if w > 0]
        height = [h for h in self._height[split] if h > 0]
        return width, height
