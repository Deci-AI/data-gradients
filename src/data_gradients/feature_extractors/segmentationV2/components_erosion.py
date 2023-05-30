import cv2
import pandas as pd
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import Hist2DPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor
from data_gradients.batch_processors.preprocessors import contours


@register_feature_extractor()
class SegmentationComponentsErosion(AbstractFeatureExtractor):
    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):
        label = sample.mask.transpose(1, 2, 0).astype(np.uint8)
        label = cv2.morphologyEx(label, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if len(label.shape) == 2:
            label = label[..., np.newaxis]
        eroded_contours = contours.get_contours(label.transpose(2, 0, 1))

        for j, class_channel in enumerate(sample.contours):
            for contour in class_channel:
                class_id = contour.class_id
                class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "eroded": "Without Erosion",
                        "class_name": class_name,
                    }
                )

        for j, class_channel in enumerate(eroded_contours):
            for contour in class_channel:
                class_id = contour.class_id
                class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "eroded": "With Erosion",
                        "class_name": class_name,
                    }
                )

        # FIXME: I dont think this is what we want to do

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = Hist2DPlotOptions(
            x_label_key="eroded",
            x_label_name="Number of Components",
            title=self.title,
            x_ticks_rotation=None,
            labels_key="split",
            individual_plots_key="split",
        )

        json = dict(df.describe())

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Components Erosion."

    @property
    def description(self) -> str:
        return (
            "An assessment of object stability under morphological opening - erosion followed by dilation. "
            "When a lot of components are small then the number of components decrease which means we might have "
            "noise in our annotations (i.e 'sprinkles')."
        )
        # FIXME: Can this also lead to increase of components, when breaking existing component into 2?
