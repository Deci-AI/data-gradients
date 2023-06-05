import numpy as np
from collections import defaultdict
from typing import Dict, List
from abc import ABC, abstractmethod

from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.visualize.plot_options import FigureRenderer
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.visualize.images import stack_split_images_to_fig, combine_images


@register_feature_extractor()
class AbstractSampleVisualization(AbstractFeatureExtractor, ABC):
    def __init__(self, n_samples_per_split: int = 9, n_cols_per_split: int = 3, stack_splits_vertically: bool = True):
        """
        :param n_samples_per_split:     Number of samples to visualize per split
        :param n_cols_per_split:        Number of columns to use per split
        :param stack_splits_vertically: Specifies whether to display the splits vertically stacked.
                                        If set to False, the splits will be shown side by side
        """
        self.n_samples_per_split = n_samples_per_split
        self.n_cols_per_split = n_cols_per_split
        self.stack_splits_vertically = stack_splits_vertically
        self.images_per_split: Dict[str, List[np.ndarray]] = defaultdict(list)

    def update(self, sample: ImageSample):
        split_images = self.images_per_split[sample.split]

        if len(split_images) < self.n_samples_per_split:
            image = self._prepare_sample_visualization(sample=sample)
            split_images.append(image)

    @abstractmethod
    def _prepare_sample_visualization(self, sample: ImageSample) -> np.ndarray:
        """Combine image and label to a single image.

        :param sample: Input image sample
        :return: The preprocessed image tensor.
        """
        ...

    def aggregate(self) -> Feature:
        plot_options = FigureRenderer(title=self.title)

        # Generate a single image per split
        combined_images_per_split = {
            split: combine_images(split_images, n_cols=self.n_cols_per_split, row_figsize=(10, 2.5)) for split, split_images in self.images_per_split.items()
        }

        # Generate a single figure
        fig = stack_split_images_to_fig(
            image_per_split=combined_images_per_split,
            split_figsize=(10, 6),
            tight_layout=True,
            stack_vertically=self.stack_splits_vertically,
        )

        feature = Feature(data=fig, plot_options=plot_options, json={})
        return feature

    @property
    def title(self) -> str:
        return "Visualization of Samples"

    @property
    def description(self) -> str:
        return (
            f"Visualization of {self.n_samples_per_split} samples per split. "
            f"This can be useful to make sure the mapping of class_names to class_ids is done correctly, "
            f"but also to get a better understanding of what your dataset is made of.."
        )