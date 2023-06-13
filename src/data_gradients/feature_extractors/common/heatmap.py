from typing import Tuple, Dict
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import FigureRenderer
from data_gradients.feature_extractors.abstract_feature_extractor import Feature, AbstractFeatureExtractor
from data_gradients.visualize.images import combine_images_per_split_per_class


class BaseClassHeatmap(AbstractFeatureExtractor, ABC):
    def __init__(self, n_classes_to_show: int = 12, n_cols: int = 2, heatmap_dim: Tuple[int, int] = (200, 200)):
        """
        :param n_classes_to_show:   The `n_classes_to_show` classes that are the most represented in the dataset will be shown.
        :param heatmap_dim:         Dimensions of the heatmap. Increase for more resolution, at the expense of processing speed
        """
        self.heatmap_dim = heatmap_dim
        self.n_classes_to_show = n_classes_to_show
        self.n_cols = n_cols

        self.class_names = []
        self.heatmaps_per_split: Dict[str, np.ndarray] = {}  # Each heatmap should be of shape (n_class, heatmap_dim[0], heatmap_dim[1])

    @abstractmethod
    def update(self, sample: SegmentationSample):
        ...

    def aggregate(self) -> Feature:
        # Select top k heatmaps by appearance
        split_count = sum(split_heatmap.sum(axis=(1, 2)) for split_heatmap in self.heatmaps_per_split.values())
        most_used_class_ids = (-split_count).argsort()[: self.n_classes_to_show]

        # Normalize (0-1)
        normalized_heatmaps_per_split_per_cls = defaultdict(dict)
        for split, heatmaps in self.heatmaps_per_split.items():
            for class_id, heatmap in enumerate(heatmaps):
                if class_id in most_used_class_ids:
                    class_name = self.class_names[class_id]
                    normalized_heatmaps_per_split_per_cls[class_name][split] = (255 * (heatmap / (heatmap.max() + 1e-6))).astype(np.uint8)

        fig = combine_images_per_split_per_class(images_per_split_per_class=normalized_heatmaps_per_split_per_cls, n_cols=self.n_cols)
        plot_options = FigureRenderer(title=self.title)
        json = {class_name: "No Data" for class_name in normalized_heatmaps_per_split_per_cls.keys()}

        return Feature(data=fig, plot_options=plot_options, json=json)