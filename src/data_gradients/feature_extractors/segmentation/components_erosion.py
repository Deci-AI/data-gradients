import cv2
from typing import Tuple
import pandas as pd
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import KDEPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.batch_processors.preprocessors import contours


@register_feature_extractor()
class SegmentationComponentsErosion(AbstractFeatureExtractor):
    def __init__(self):
        self.kernel_shape = (3, 3)
        self.data = []

    def update(self, sample: SegmentationSample):
        opened_mask = self.apply_mask_opening(mask=sample.mask, kernel_shape=self.kernel_shape)
        contours_after_opening = contours.get_contours(opened_mask)

        if sample.contours:
            n_components_without_opening = sum(1 for class_channel in sample.contours for _contour in class_channel)
            n_components_after_opening = sum(1 for class_channel in contours_after_opening for _contour in class_channel)

            increase_of_n_components = n_components_after_opening - n_components_without_opening
            percent_change_of_n_components = 100 * (increase_of_n_components / n_components_without_opening)
        else:
            percent_change_of_n_components = 0

        self.data.append(
            {
                "split": sample.split,
                "percent_change_of_n_components": percent_change_of_n_components,
            }
        )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = KDEPlotOptions(
            x_label_key="percent_change_of_n_components",
            x_label_name="Increase of number of components in %",
            title=self.title,
            x_ticks_rotation=None,
            labels_key="split",
            common_norm=False,
            fill=True,
            sharey=True,
        )

        json = dict(
            train=dict(df[df["split"] == "train"]["percent_change_of_n_components"].describe()),
            val=dict(df[df["split"] == "val"]["percent_change_of_n_components"].describe()),
        )

        return Feature(data=df, plot_options=plot_options, json=json)

    @property
    def title(self) -> str:
        return "Objects Stability to Erosion"

    @property
    def description(self) -> str:
        return (
            "Assessment of object stability under morphological opening - erosion followed by dilation. "
            "When a lot of components are small then the number of components decrease which means we might have "
            "noise in our annotations (i.e 'sprinkles')."
        )
        # FIXME: Can this also lead to increase of components, when breaking existing component into 2?

    def apply_mask_opening(self, mask: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
        """Opening is just another name of erosion followed by dilation.

        It is useful in removing noise, as we explained above. Here we use the function, cv2.morphologyEx(). See [Official OpenCV documentation](
        https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)

        :param mask:            Mask to open in shape [N, H, W]
        :param kernel_shape:    Shape of the kernel used for Opening (Eroded + Dilated)
        :return:                Opened (Eroded + Dilated) mask in shape [N, H, W]
        """
        masks = mask.transpose((1, 2, 0)).astype(np.uint8)
        masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN, np.ones(kernel_shape, np.uint8))
        return masks.transpose((2, 0, 1))
