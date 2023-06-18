from typing import Tuple
import numpy as np
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.feature_extractors.common.heatmap import BaseClassHeatmap
from data_gradients.utils.detection import scale_bboxes


@register_feature_extractor()
class DetectionClassHeatmap(BaseClassHeatmap):
    def __init__(self, n_rows: int = 12, n_cols: int = 2, heatmap_shape: Tuple[int, int] = (200, 200)):
        """
        :param n_rows:          How many rows per split.
        :param n_cols:          How many columns per split.
        :param heatmap_shape:   Heatmap, in (H, W) format. Increase for more resolution, at the expense of processing speed.
        """
        super().__init__(n_rows=n_rows, n_cols=n_cols, heatmap_shape=heatmap_shape)

    def update(self, sample: DetectionSample):

        if not self.class_names:
            self.class_names = sample.class_names

        original_shape = sample.image.shape[:2]
        bboxes_xyxy = scale_bboxes(old_shape=original_shape, new_shape=self.heatmap_shape, bboxes_xyxy=sample.bboxes_xyxy)

        split_heatmap = self.heatmaps_per_split.get(sample.split, np.zeros((len(sample.class_names), *self.heatmap_shape)))

        for class_id, (x1, y1, x2, y2) in zip(sample.class_ids, bboxes_xyxy):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            split_heatmap[class_id, y1:y2, x1:x2] += 1

        self.heatmaps_per_split[sample.split] = split_heatmap

    @property
    def title(self) -> str:
        return "Bounding Boxes Density"

    @property
    def description(self) -> str:
        return (
            "The heatmap represents areas of high object density within the images, providing insights into the spatial distribution of objects. "
            "By examining the heatmap, you can quickly identify if objects are predominantly concentrated in specific regions or if they are evenly "
            "distributed throughout the scene. This information can serve as a heuristic to assess if the objects are positioned appropriately "
            "within the expected areas of interest."
        )

    @property
    def notice(self) -> str:
        return (
            f"Only the {self.n_cols * self.n_rows} classes with highest density are shown.<br/>"
            f"You can increase the number of classes by changing `n_cols` and `n_rows` in the configuration file."
        )
