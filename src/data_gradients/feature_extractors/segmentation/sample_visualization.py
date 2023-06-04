import cv2
import numpy
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import SegmentationSample, ImageChannelFormat
from data_gradients.visualize.plot_options import ImagesRenderer
from data_gradients.feature_extractors.abstract_feature_extractor import Feature


@register_feature_extractor()
class SegmentationSampleVisualization(AbstractFeatureExtractor):
    def __init__(self, n_samples_to_visualize: int = 12, n_cols: int = 3, stack_mask_horizontally: bool = False):
        """
        :param n_samples_to_visualize:  Number of samples to visualize.
        :param n_cols:                  Number of columns in the grid.
        :param stack_mask_horizontally: Whether to show the image and the mask next one another. If False, the mask will be shown below the image instead.
        """
        self.n_samples_to_visualize = n_samples_to_visualize
        self.n_cols = n_cols
        self.stack_mask_horizontally = stack_mask_horizontally
        self.samples = []

    def update(self, sample: SegmentationSample):

        if len(self.samples) < self.n_samples_to_visualize:
            image = self.prepare_segmentation_image(
                image=sample.image,
                labels=sample.mask,
                image_format=sample.image_format,
                stack_mask_horizontally=self.stack_mask_horizontally,
            )
            self.samples.append(image)

    def aggregate(self) -> Feature:
        plot_options = ImagesRenderer(title=self.title, n_cols=self.n_cols)
        feature = Feature(data=self.samples, plot_options=plot_options, json={})
        return feature

    @property
    def title(self) -> str:
        return "Visualization of Samples"

    @property
    def description(self) -> str:
        return (
            f"Visualization of {self.n_samples_to_visualize} random samples from the dataset. "
            f"This can be useful to see how your dataset looks like which can be valuable to understand following features."
        )

    @staticmethod
    def prepare_segmentation_image(
        image: numpy.ndarray,
        labels: numpy.ndarray,
        image_format: ImageChannelFormat,
        stack_mask_horizontally: bool,
    ) -> numpy.ndarray:
        """Combine image and label to a single image.

        :param image:           Input image tensor
        :param labels:          Input Labels
        :param stack_mask_horizontally:  Whether to show the image and the mask next one another. If False, the mask will be shown below the image instead.
        :return:                The preprocessed image tensor.
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
        if stack_mask_horizontally:
            result = np.hstack((image, normalized_labels))
        else:
            result = np.vstack((image, normalized_labels))

        return result.astype(np.uint8)
