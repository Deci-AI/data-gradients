import cv2
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import DetectionSample, ImageChannelFormat
from data_gradients.visualize.plot_options import ImagesRenderer
from data_gradients.visualize.detection import draw_bboxes
from data_gradients.feature_extractors.abstract_feature_extractor import Feature


@register_feature_extractor()
class DetectionSampleVisualization(AbstractFeatureExtractor):
    def __init__(self, n_samples_to_visualize: int = 8, n_cols: int = 2):
        """
        :param n_samples_to_visualize:  Number of samples to visualize.
        :param n_cols:                  Number of columns in the grid.
        """
        self.n_samples_to_visualize = n_samples_to_visualize
        self.n_cols = n_cols
        self.samples = []

    def update(self, sample: DetectionSample):

        if len(self.samples) < self.n_samples_to_visualize:
            class_names = np.array([sample.class_names[class_id] if sample.class_names is not None else str(class_id) for class_id in sample.class_ids])
            image = self.prepare_detection_image(image=sample.image, class_names=class_names, bboxes_xyxy=sample.bboxes_xyxy, image_format=sample.image_format)
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
    def prepare_detection_image(
        image: np.ndarray,
        class_names: np.ndarray,
        bboxes_xyxy: np.ndarray,
        image_format: ImageChannelFormat,
    ) -> np.ndarray:
        """Combine image and label to a single image.

        :param image:           Input image tensor
        :param bboxes_xyxy:     np.ndarray of shape [N, 4] (X, Y, X, Y)
        :param class_names:     np.ndarray of shape [N, ]
        :param image_format:    Instance of ImageChannelFormat
        :return:                The preprocessed image.
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

        result = draw_bboxes(image=image, class_names=class_names, bboxes_xyxy=bboxes_xyxy)
        return result.astype(np.uint8)
