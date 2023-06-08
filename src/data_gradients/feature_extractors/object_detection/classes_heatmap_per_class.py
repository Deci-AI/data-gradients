from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.feature_extractors.common.heatmap import BaseClassHeatmap
from data_gradients.utils.detection import scale_bboxes


@register_feature_extractor()
class DetectionClassHeatmap(BaseClassHeatmap):
    def update(self, sample: DetectionSample):

        original_size = sample.image.shape[:2]
        bboxes_xyxy = scale_bboxes(old_size=original_size, new_size=self.heatmap_dim, bboxes_xyxy=sample.bboxes_xyxy)

        for class_id, (x1, y1, x2, y2) in zip(sample.class_ids, bboxes_xyxy):
            class_name = sample.class_names[class_id]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            self.heatmaps_per_split_per_cls[class_name][sample.split][y1:y2, x1:x2] += 1
            self.count_class_appearance[class_name] += 1
