from typing import Tuple
import cv2
import numpy as np
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.image_processing import resize_in_chunks
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.common.heatmap import BaseClassHeatmap


@register_feature_extractor()
class SegmentationClassHeatmap(BaseClassHeatmap):
    def __init__(self, n_rows: int = 12, n_cols: int = 2, heatmap_dim: Tuple[int, int] = (200, 200)):
        """
        :param n_rows:      How many rows per split.
        :param n_cols:      How many columns per split.
        :param heatmap_dim: Dimensions of the heatmap. Increase for more resolution, at the expense of processing speed.
        """
        super().__init__(n_rows=n_rows, n_cols=n_cols, heatmap_dim=heatmap_dim)

    def update(self, sample: SegmentationSample):

        if not self.class_names:
            self.class_names = sample.class_names

        # Objects are resized to a fix size
        mask = sample.mask.transpose((1, 2, 0))
        resized_masks = resize_in_chunks(img=mask, size=self.heatmap_dim, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        resized_masks = resized_masks.transpose((2, 0, 1))

        split_heatmap = self.heatmaps_per_split.get(sample.split, np.zeros((len(sample.class_names), *self.heatmap_dim)))
        split_heatmap += resized_masks
        self.heatmaps_per_split[sample.split] = split_heatmap

    @property
    def title(self) -> str:
        return "Heatmap of Segmentation Masks"

    @property
    def description(self) -> str:
        return (
            "Show the areas of high density of Bounding Boxes. This can be useful to understand if the objects are positioned in the right area.\n"
            f"Note that only top {self.n_cols * self.n_rows} classes are shown. "
            f" You can increase the number of classes by setting `SegmentationClassHeatmap` with `n_classes_to_show`"
        )
