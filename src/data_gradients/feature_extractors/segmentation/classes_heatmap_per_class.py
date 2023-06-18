from typing import Tuple
import cv2
import numpy as np
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.image_processing import resize_in_chunks
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.common.heatmap import BaseClassHeatmap


@register_feature_extractor()
class SegmentationClassHeatmap(BaseClassHeatmap):
    def __init__(self, n_rows: int = 12, n_cols: int = 2, heatmap_shape: Tuple[int, int] = (200, 200)):
        """
        :param n_rows:          How many rows per split.
        :param n_cols:          How many columns per split.
        :param heatmap_shape:   Heatmap, in (H, W) format. Increase for more resolution, at the expense of processing speed.
        """
        super().__init__(n_rows=n_rows, n_cols=n_cols, heatmap_shape=heatmap_shape)

    def update(self, sample: SegmentationSample):

        if not self.class_names:
            self.class_names = sample.class_names

        # Objects are resized to a fix size
        mask = sample.mask.transpose((1, 2, 0))

        target_size = self.heatmap_shape[1], self.heatmap_shape[0]
        resized_masks = resize_in_chunks(img=mask.astype(np.uint8), size=target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        resized_masks = resized_masks.transpose((2, 0, 1))

        split_heatmap = self.heatmaps_per_split.get(sample.split, np.zeros((len(sample.class_names), *self.heatmap_shape)))
        split_heatmap += resized_masks
        self.heatmaps_per_split[sample.split] = split_heatmap

    @property
    def title(self) -> str:
        return "Objects Density"

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
