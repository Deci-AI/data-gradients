import matplotlib.colors as mcolors
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.common.sample_visualization import AbstractSampleVisualization
from data_gradients.utils.data_classes.data_samples import SegmentationSample

from data_gradients.visualize.detection.utils import generate_gray_color_mapping


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
        if sample.image_as_rgb is None:
            raise RuntimeError(f"`{self.__class__.__name__}` not compatible with Image format `{sample.image_channels.__class__.__name__}`")

        image = sample.image_as_rgb
        mask = sample.mask

        class_ids = list(sample.class_names.keys())

        # Create a color map using the generate_gray_color_mapping function
        colors = generate_gray_color_mapping(len(class_ids) + 1)  # generated colors for each class
        cmap = mcolors.ListedColormap(colors)

        # Map class IDs to color map index
        mask_mapped = np.zeros_like(mask, dtype=int)
        for idx, c_id in enumerate(class_ids):
            mask_mapped[mask == c_id] = idx + 1  # +1 because 0 is reserved for background (idx=-1)

        # Convert mask_mapped to RGB using the colormap
        mask_rgb = (cmap(mask_mapped)[:, :, :3] * 255).astype(np.uint8)

        # Stack the image and label color map horizontally or vertically
        if self.stack_mask_vertically:
            result = np.vstack((image, mask_rgb))
        else:
            result = np.hstack((image, mask_rgb))

        return result.astype(np.uint8)
