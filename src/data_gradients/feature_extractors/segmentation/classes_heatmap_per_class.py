import cv2
import numpy as np
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.common.heatmap import BaseClassHeatmap


@register_feature_extractor()
class SegmentationClassHeatmap(BaseClassHeatmap):
    def __init__(self):
        super().__init__()

    def update(self, sample: SegmentationSample):

        if not self.class_names:
            self.class_names = sample.class_names

        # Objects are resized to a fix size
        mask = sample.mask.transpose((1, 2, 0))
        resized_masks = cv2.resize(src=mask, dsize=self.heatmap_dim, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
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
            f"Note that only top {self.n_classes_to_show} classes are shown. "
            f" You can increase the number of classes by setting `SegmentationClassHeatmap` with `n_classes_to_show`"
        )
