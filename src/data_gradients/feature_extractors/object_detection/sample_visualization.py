import cv2
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.common.sample_visualization import AbstractSampleVisualization
from data_gradients.utils.data_classes.data_samples import DetectionSample, ImageChannelFormat
from data_gradients.visualize.detection import draw_bboxes


@register_feature_extractor()
class DetectionSampleVisualization(AbstractSampleVisualization):
    def __init__(self, n_rows: int = 4, n_cols: int = 2, stack_splits_vertically: bool = True):
        """
        :param n_rows:     Number of rows to use per split
        :param n_cols:                  Number of columns to use per split
        :param stack_splits_vertically: Specifies whether to display the splits vertically stacked.
                                        If set to False, the splits will be shown side by side
        """
        super().__init__(n_rows=n_rows, n_cols=n_cols, stack_splits_vertically=stack_splits_vertically)

    def _prepare_sample_visualization(self, sample: DetectionSample) -> np.ndarray:
        """Combine image and label to a single image.

        :param sample: Input image sample
        :return: The preprocessed image tensor.
        """

        if sample.image_format == ImageChannelFormat.RGB:
            image = sample.image
        elif sample.image_format == ImageChannelFormat.BGR:
            image = cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)
        elif sample.image_format == ImageChannelFormat.GRAYSCALE:
            image = cv2.cvtColor(sample.image, cv2.COLOR_GRAY2RGB)
        elif sample.image_format == ImageChannelFormat.UNKNOWN:
            image = sample.image
        else:
            raise ValueError(f"Unknown image format {sample.image_format}")

        result = draw_bboxes(image=image, bboxes_xyxy=sample.bboxes_xyxy, bboxes_ids=sample.class_ids, class_names=sample.class_names)
        return result.astype(np.uint8)
