import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationBoundingBoxArea(AbstractFeatureExtractor):
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
                class_name = sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "class_name": class_name,
                        "class_id": class_id,
                        "bbox_area": 100 * (contour.bbox_area / image_area),
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        max_area = min(100, df["bbox_area"].max())
        plot_options = ViolinPlotOptions(
            x_label_key="bbox_area",
            x_label_name="Bounding Box Area (in % of image)",
            y_label_key="class_name",
            y_label_name="Class",
            order_key="class_id",
            title=self.title,
            x_lim=(0, max_area),
            x_ticks_rotation=None,
            labels_key="split",
            bandwidth=0.4,
        )
        json = dict(train=dict(df[df["split"] == "train"]["bbox_area"].describe()), val=dict(df[df["split"] == "val"]["bbox_area"].describe()))

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Object Area"

    @property
    def description(self) -> str:
        return (
            "This graph shows the distribution of object area for each class. "
            "This can highlight distribution gap in object size between the training and validation splits, which can harm the model performance. \n"
            "Another thing to keep in mind is that having too many very small objects may indicate that your are down sizing your original image to a "
            "low resolution that is not appropriate for your objects."
        )
