import cv2
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.feature_extractors.common.heatmap import BaseClassHeatmap


@register_feature_extractor()
class SegmentationClassHeatmap(BaseClassHeatmap):
    def update(self, sample: SegmentationSample):

        # Objects are resized to a fix size
        mask = sample.mask.transpose((1, 2, 0))
        resized_masks = cv2.resize(src=mask, dsize=self.heatmap_dim, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        resized_masks = resized_masks.transpose((2, 0, 1))

        for class_id, class_name in enumerate(sample.class_names):
            self.heatmaps_per_split_per_cls[class_name][sample.split] += resized_masks[class_id, :, :]
            self.count_class_appearance[class_name] += 1
