import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import SegmentationSample, ImageChannelFormat
from data_gradients.visualize.plot_options import FigureRenderer
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.visualize.images import stack_split_images_to_fig, combine_images


@register_feature_extractor()
class SegmentationSampleVisualization(AbstractFeatureExtractor):
    def __init__(self, n_samples_per_split: int = 9, n_cols: int = 3, stack_splits_vertically: bool = True, stack_mask_vertically: bool = True):
        """
        :param n_samples_per_split:     Number of samples to visualize
        :param n_cols:                  Number of columns in the grid
        :param stack_splits_vertically: Specifies whether to display the splits vertically stacked.
                                        If set to False, the splits will be shown side by side
        :param stack_mask_vertically:   Specifies whether to display the image and the mask vertically stacked.
                                        If set to False, the mask will be shown side by side
        """
        self.n_samples_per_split = n_samples_per_split
        self.n_cols = n_cols
        self.stack_splits_vertically = stack_splits_vertically
        self.stack_mask_vertically = stack_mask_vertically
        self.images_per_split: Dict[str, List[np.ndarray]] = defaultdict(list)

    def update(self, sample: SegmentationSample):
        split_images = self.images_per_split[sample.split]

        if len(split_images) < self.n_samples_per_split:
            image = self._prepare_segmentation_image(
                image=sample.image,
                labels=sample.mask,
                image_format=sample.image_format,
                stack_mask_vertically=self.stack_mask_vertically,
            )
            split_images.append(image)

    def aggregate(self) -> Feature:
        plot_options = FigureRenderer(title=self.title)

        # Generate a single image per split
        combined_images_per_split = {
            split: combine_images(split_images, n_cols=self.n_cols, row_figsize=(10, 2.5)) for split, split_images in self.images_per_split.items()
        }

        # Generate a single image
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

    @staticmethod
    def _prepare_segmentation_image(
        image: np.ndarray,
        labels: np.ndarray,
        image_format: ImageChannelFormat,
        stack_mask_vertically: bool,
    ) -> np.ndarray:
        """Combine image and label to a single image.

        :param image:                 Input image tensor
        :param labels:                Input Labels
        :param stack_mask_vertically: Whether to show the image and the mask next one another. If True, the mask will be shown below the image instead.
        :return: The preprocessed image tensor.
        """

        if image_format == ImageChannelFormat.RGB:
            image = image
        elif image_format == ImageChannelFormat.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_format == ImageChannelFormat.GRAYSCALE:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image_format == ImageChannelFormat.UNKNOWN:
            image = image
        else:
            raise ValueError(f"Unknown image format {image_format}")

        # Onehot to categorical labels
        categorical_labels = np.argmax(labels, axis=0)

        # Normalize the labels to the range [0, 255]
        normalized_labels = np.ceil((categorical_labels * 255) / np.max(categorical_labels))
        normalized_labels = normalized_labels[:, :, np.newaxis].repeat(3, axis=-1)

        # Stack the image and label color map horizontally or vertically
        if stack_mask_vertically:
            result = np.vstack((image, normalized_labels))
        else:
            result = np.hstack((image, normalized_labels))

        return result.astype(np.uint8)
