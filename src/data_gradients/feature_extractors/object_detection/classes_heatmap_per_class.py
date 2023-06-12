import numpy as np
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.feature_extractors.common.heatmap import BaseClassHeatmap
from data_gradients.utils.detection import scale_bboxes


@register_feature_extractor()
class DetectionClassHeatmap(BaseClassHeatmap):
    def update(self, sample: DetectionSample):

        if not self.class_names:
            self.class_names = sample.class_names

        original_size = sample.image.shape[:2]
        bboxes_xyxy = scale_bboxes(old_size=original_size, new_size=self.heatmap_dim, bboxes_xyxy=sample.bboxes_xyxy)

        split_heatmap = self.heatmaps_per_split.get(sample.split, np.zeros((len(sample.class_names), *self.heatmap_dim)))

        for class_id, (x1, y1, x2, y2) in zip(sample.class_ids, bboxes_xyxy):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            split_heatmap[class_id, y1:y2, x1:x2] += 1

        self.heatmaps_per_split[sample.split] = split_heatmap

    @property
    def title(self) -> str:
        return "Heatmap of Bounding Boxes"

    @property
    def description(self) -> str:
        return (
            "Show the areas of high density of Bounding Boxes. This can be useful to understand if the objects are positioned in the right area.\n"
            f"Note that only top {self.n_classes_to_show} classes are shown. "
            f" You can increase the number of classes by setting `DetectionClassHeatmap` with `n_classes_to_show`"
        )
