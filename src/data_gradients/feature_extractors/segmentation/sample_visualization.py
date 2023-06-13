import cv2
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.common.sample_visualization import AbstractSampleVisualization
from data_gradients.utils.data_classes.data_samples import SegmentationSample, ImageChannelFormat


@register_feature_extractor()
class SegmentationSampleVisualization(AbstractSampleVisualization):
    def __init__(self, n_rows: int = 9, n_cols: int = 3, stack_splits_vertically: bool = True, stack_mask_vertically: bool = True):
        """
        :param n_rows:                  Number of rows to use per split
        :param n_cols:                  Number of columns to use per split
        :param stack_splits_vertically: Specifies whether to display the splits vertically stacked.
                                        If set to False, the splits will be shown side by side
        :param stack_mask_vertically:   Specifies whether to display the image and the mask vertically stacked.
                                        If set to False, the mask will be shown side by side
        """
        super().__init__(n_rows=n_rows, n_cols=n_cols, stack_splits_vertically=stack_splits_vertically)
        self.stack_mask_vertically = stack_mask_vertically

    def _prepare_sample_visualization(self, sample: SegmentationSample) -> np.ndarray:
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

        # Onehot to categorical labels
        categorical_labels = np.argmax(sample.mask, axis=0)

        # Normalize the labels to the range [0, 255]
        normalized_labels = np.ceil((categorical_labels * 255) / np.max(categorical_labels))
        normalized_labels = normalized_labels[:, :, np.newaxis].repeat(3, axis=-1)

        # Stack the image and label color map horizontally or vertically
        if self.stack_mask_vertically:
            result = np.vstack((image, normalized_labels))
        else:
            result = np.hstack((image, normalized_labels))

        return result.astype(np.uint8)
