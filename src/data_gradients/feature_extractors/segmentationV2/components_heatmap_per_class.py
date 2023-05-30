import cv2
from typing import List, Optional, Dict
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import ImagePlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationComponentHeatmap(AbstractFeatureExtractor):
    def __init__(self):
        self.heatmap_dim = (1000, 1000)
        self.kernel_shape = (3, 3)
        self.heatmaps_per_split_per_cls: Dict[str, Dict[str, np.ndarray]] = {}
        self.class_names: Optional[List[str]] = None

    def new_heatmap(self):
        return np.zeros(self.heatmap_dim, dtype=np.uint8)

    def update(self, sample: SegmentationSample):

        if self.class_names is None:
            self.class_names = sample.class_names

        # Making sure all the masks are the same size (100, 100) for visualization.
        mask = sample.mask.transpose((1, 2, 0))
        resized_masks = cv2.resize(src=mask, dsize=self.heatmap_dim, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        resized_masks = resized_masks.transpose((2, 0, 1))

        for class_id, mask in enumerate(resized_masks):
            class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]

            if class_name not in self.heatmaps_per_split_per_cls:
                self.heatmaps_per_split_per_cls[class_name] = {}

            if sample.split not in self.heatmaps_per_split_per_cls[class_name]:
                self.heatmaps_per_split_per_cls[class_name][sample.split] = self.new_heatmap()

            self.heatmaps_per_split_per_cls[class_name][sample.split] += mask

    def aggregate(self) -> Feature:
        plot_options = ImagePlotOptions(title=self.title, tight_layout=True)

        cleaned_heatmaps_per_split_per_cls = {}
        for class_name, heatmaps_per_split in self.heatmaps_per_split_per_cls.items():
            cleaned_heatmaps_per_split_per_cls[class_name] = {}
            for split, heatmap in heatmaps_per_split.items():
                cleaned_heatmaps_per_split_per_cls[class_name][split] = (255 * (heatmap / heatmap.max())).astype(np.uint8)

        json = {class_name: "No Data" for class_name in cleaned_heatmaps_per_split_per_cls.keys()}
        return Feature(data=cleaned_heatmaps_per_split_per_cls, plot_options=plot_options, json=json)

    @property
    def title(self) -> str:
        return "Heatmap of Class"

    @property
    def description(self) -> str:
        return "Show the areas of high density of components. This can be useful to understand if the objects are positioned in the right area."
