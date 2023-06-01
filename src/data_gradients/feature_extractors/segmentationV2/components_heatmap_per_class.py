import cv2
from typing import Tuple
import numpy as np
from collections import defaultdict

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import ImageHeatmapPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationComponentHeatmap(AbstractFeatureExtractor):
    def __init__(self, n_classes_to_show: int = 12, heatmap_dim: Tuple[int, int] = (200, 200)):
        """
        :param n_classes_to_show:   The `n_classes_to_show` classes that are the most represented in the dataset will be shown.
        :param heatmap_dim:         Dimensions of the heatmap. Increase for more resolution, at the expense of processing speed
        """
        self.heatmap_dim = heatmap_dim
        self.n_classes_to_show = n_classes_to_show

        self.heatmaps_per_split_per_cls = defaultdict(lambda: defaultdict(lambda: np.zeros(self.heatmap_dim, dtype=np.uint8)))
        self.count_class_appearance = defaultdict(lambda: 0)

    def update(self, sample: SegmentationSample):

        # Making sure all the masks are the same size (100, 100) for visualization.
        mask = sample.mask.transpose((1, 2, 0))
        resized_masks = cv2.resize(src=mask, dsize=self.heatmap_dim, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        resized_masks = resized_masks.transpose((2, 0, 1))

        for class_id, mask in enumerate(resized_masks):
            class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]
            self.heatmaps_per_split_per_cls[class_name][sample.split] += mask
            self.count_class_appearance[class_name] += 1

    def aggregate(self) -> Feature:

        # Select top k heatmaps by appearance
        most_used_heatmaps_tuples = sorted(self.count_class_appearance.items(), key=lambda x: x[1], reverse=True)
        most_used_classes = [class_name for (class_name, n_used_pixels) in most_used_heatmaps_tuples][: self.n_classes_to_show]

        # Normalize (0-1)
        cleaned_heatmaps_per_split_per_cls = {}
        for class_name, heatmaps_per_split in self.heatmaps_per_split_per_cls.items():
            if class_name in most_used_classes:
                cleaned_heatmaps_per_split_per_cls[class_name] = {}
                for split, heatmap in heatmaps_per_split.items():
                    cleaned_heatmaps_per_split_per_cls[class_name][split] = (255 * (heatmap / heatmap.max())).astype(np.uint8)

        plot_options = ImageHeatmapPlotOptions(title=self.title)
        json = {class_name: "No Data" for class_name in cleaned_heatmaps_per_split_per_cls.keys()}
        return Feature(data=cleaned_heatmaps_per_split_per_cls, plot_options=plot_options, json=json)

    @property
    def title(self) -> str:
        return "Heatmap of Object Density"

    @property
    def description(self) -> str:
        return "Show the areas of high density of components. This can be useful to understand if the objects are positioned in the right area."
