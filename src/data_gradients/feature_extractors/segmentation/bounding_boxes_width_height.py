import numpy as np
import torch

from data_gradients.utils import SegBatchData
from data_gradients.feature_extractors.feature_extractor_abstract import FeatureExtractorAbstract
from data_gradients.utils.data_classes.extractor_results import HeatMapResults


class WidthHeight(FeatureExtractorAbstract):
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
            for j, class_channel in enumerate(image_contours):
                height, width = data.labels[i][j].shape
                for contour in class_channel:
                    self._width[data.split].append(contour.w / width)
                    self._height[data.split].append(contour.h / height)

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
        Resultsjson_values = quantized_heat_map.tolist()
        Resultskeys = ["Width", "Height"]
        return results

    def _process_data(self, split: str):
        width = [w for w in self._width[split] if w > 0]
        height = [h for h in self._height[split] if h > 0]
        return width, height
