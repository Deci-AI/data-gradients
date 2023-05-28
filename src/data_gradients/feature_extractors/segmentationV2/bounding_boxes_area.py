import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import ViolinPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class BoundingBoxAreaFeatureExtractor(AbstractFeatureExtractor):
    """
    Semantic Segmentation task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """

    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):
        image_area = sample.image.shape[0] * sample.image.shape[1]
        for class_channel in sample.contours:
            for contour in class_channel:
                class_id = contour.class_id
                class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "class_name": class_name,
                        "bbox_area": 100 * contour.bbox_area / image_area,
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)
        title = "Distribution of Bounding Boxes Area per Class"

        plot_options = ViolinPlotOptions(
            x_label_key="bbox_area",
            x_label_name="Bound Box Area (in % of image)",
            y_label_key="class_name",
            y_label_name="Class",
            title=title,
            x_ticks_rotation=None,
            labels_key="split",
        )
        json = ["coming soon..."]  # TODO: define what we want here. We might not want all the data, but in that case how to split it?

        feature = Feature(
            title=title,
            description="Coming soon...",
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature
