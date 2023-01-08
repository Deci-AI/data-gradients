import numpy as np

from src.logging.logger_utils import write_heatmap_plot, create_json_object
from src.preprocess import contours
from src.utils import SegBatchData
from src.feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from src.utils.data_classes.extractor_results import HeatMapResults


class ComponentsCenterOfMass(SegmentationFeatureExtractorAbstract):
    """
    Semantic Segmentation task feature extractor -
    Get all X, Y positions of center of mass of every object in every image for every class.
    Plot those X, Y positions as a heat-map
    """
    def __init__(self, num_classes, ignore_labels):
        super().__init__()
        keys = [int(i) for i in range(0, num_classes + len(ignore_labels)) if i not in ignore_labels]
        self._hist = {'train': {k: {'x': list(), 'y': list()} for k in keys},
                      'val': {k: {'x': list(), 'y': list()} for k in keys}}

        self.num_axis = (1, 2)

    def _execute(self, data: SegBatchData):
        for i, image_contours in enumerate(data.contours):
            for j, cls_contours in enumerate(image_contours):
                unique = np.unique(data.labels[i][j])
                if not len(unique) > 1:
                    continue
                for c in cls_contours:
                    self._hist[data.split][int(np.delete(unique, 0))]['x'].append(c.center[0])
                    self._hist[data.split][int(np.delete(unique, 0))]['y'].append(c.center[1])

    def _post_process(self, split):
        # TODO: Divide each plot for a class. Need to make x, y as a dictionaries (every class..)
        x, y = self._process_data(split)
        # TODO: My thumb rules numbers
        n_bins = int(np.sqrt(len(x)) * 4)
        sigma = int(2 * (n_bins / 150))

        results = HeatMapResults(x=x,
                                 y=y,
                                 n_bins=n_bins,
                                 sigma=sigma,
                                 split=split,
                                 plot='heat-map',
                                 title=f'Center of mass average locations',
                                 x_label='X axis',
                                 y_label='Y axis'
                                 )

        quantized_heat_map, _, _ = np.histogram2d(x, y, bins=25)
        results.json_values = quantized_heat_map.tolist()
        results.keys = ["X", "Y"]
        return results

    def _process_data(self, split):
        x, y = [], []
        for val in self._hist[split].values():
            x.extend(val['x'])
            y.extend(val['y'])
        return x, y


