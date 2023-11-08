import cv2
from typing import Tuple
import pandas as pd
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import KDEPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.sample_preprocessor.utils import contours


@register_feature_extractor()
class SegmentationComponentsErosion(AbstractFeatureExtractor):
    """
    Analyzes the impact of morphological operations on the segmentation mask components within a dataset, quantifying the change in
    the number of components post-erosion.

    This feature useful for identifying and quantifying noise or small artifacts ('sprinkles') in segmentation masks,
    which may otherwise affect the performance of segmentation models.
    """

    def __init__(self):
        self.kernel_shape = (3, 3)
        self.data = []

    def update(self, sample: SegmentationSample):
        from data_gradients.utils.segmentation import mask_to_onehot

        onehot_mask = mask_to_onehot(mask_categorical=sample.mask, n_classes=max(sample.class_names.keys()))
        opened_onehot_mask = self.apply_mask_opening(onehot_mask=onehot_mask, kernel_shape=self.kernel_shape)
        opened_categorical_mask = np.argmax(opened_onehot_mask, axis=-1)

        contours_after_opening = contours.get_contours(label=opened_categorical_mask, class_ids=list(sample.class_names.keys()))

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

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
            title="Object Stability to Erosion",
            description=(
                "Assessment of object stability under morphological opening - erosion followed by dilation. "
                "When a lot of components are small then the number of components decrease which means we might have "
                "noise in our annotations (i.e 'sprinkles')."
            ),
        )
        return feature

    def apply_mask_opening(self, onehot_mask: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
        """Opening is just another name of erosion followed by dilation.

        It is useful in removing noise, as we explained above. Here we use the function, cv2.morphologyEx(). See [Official OpenCV documentation](
        https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)

        :param mask:            Mask to open in shape [N, H, W]
        :param kernel_shape:    Shape of the kernel used for Opening (Eroded + Dilated)
        :return:                Opened (Eroded + Dilated) mask in shape [N, H, W]
        """
        masks = onehot_mask.transpose((1, 2, 0)).astype(np.uint8)
        masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN, np.ones(kernel_shape, np.uint8))
        return masks.transpose((2, 0, 1))
