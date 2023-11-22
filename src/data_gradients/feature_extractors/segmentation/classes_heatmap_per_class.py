from typing import Tuple, Optional
import cv2
import numpy as np
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.image_processing import resize_in_chunks
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.common.heatmap import BaseClassHeatmap
from data_gradients.utils.segmentation import mask_to_onehot


@register_feature_extractor()
class SegmentationClassHeatmap(BaseClassHeatmap):
    """
    Provides a visual representation of object distribution across images in the dataset using heatmaps.

    It helps identify common areas where objects are frequently detected, allowing insights into potential
    biases in object placement or dataset collection.
    """

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

        # (H, W) -> (C, H, W)
        n_classes = np.max(list(sample.class_names.keys()))
        mask_onehot = mask_to_onehot(mask_categorical=sample.mask, n_classes=n_classes)
        mask_onehot = mask_onehot.transpose((1, 2, 0))  # H, W, C -> C, H, W

        target_size = self.heatmap_shape[1], self.heatmap_shape[0]
        resized_masks = resize_in_chunks(img=mask_onehot, size=target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        resized_masks = resized_masks.transpose((2, 0, 1))  # H, W, C -> C, H, W

        split_heatmap = self.heatmaps_per_split.get(sample.split, np.zeros((n_classes, *self.heatmap_shape)))
        split_heatmap += resized_masks
        self.heatmaps_per_split[sample.split] = split_heatmap

    def _generate_title(self) -> str:
        return "Objects Density"

    def _generate_description(self) -> str:
        return (
            "The heatmap represents areas of high object density within the images, providing insights into the spatial distribution of objects. "
            "By examining the heatmap, you can quickly detect whether objects are predominantly concentrated in specific regions or if they are evenly "
            "distributed throughout the scene. This information can serve as a heuristic to assess if the objects are positioned appropriately "
            "within the expected areas of interest.<br/>"
            "Note that images are resized to a square of the same dimension, which can affect the aspect ratio of objects. "
            "This is done to focus on localization of objects in the scene (e.g. top-right, center, ...) independently of the original image sizes."
        )

    def _generate_notice(self) -> Optional[str]:
        if len(self.class_names) > self.n_cols * self.n_rows:
            return (
                f"Only the {self.n_cols * self.n_rows} classes with highest density are shown.<br/>"
                f"You can increase the number of classes by changing `n_cols` and `n_rows` in the configuration file."
            )
        else:
            return None
