import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.seaborn_renderer import ViolinPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionBoundingBoxArea(AbstractFeatureExtractor):
    """
    Semantic Segmentation task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """

    def __init__(self):
        self.data = []

    def update(self, sample: DetectionSample):
        image_area = sample.image.shape[0] * sample.image.shape[1]
        for label_id, bbox_xyxy in zip(sample.labels, sample.bboxes_xyxy):
            class_name = str(label_id) if sample.class_names is None else sample.class_names[label_id]
            bbox_area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
            self.data.append(
                {
                    "split": sample.split,
                    "class_name": class_name,
                    "relative_bbox_area": 100 * (bbox_area / image_area),
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = ViolinPlotOptions(
            x_label_key="relative_bbox_area",
            x_label_name="Bound Box Area (in % of image)",
            y_label_key="class_name",
            y_label_name="Class",
            title=self.title,
            x_ticks_rotation=None,
            labels_key="split",
            bandwidth=0.4,
        )
        json = dict(df["relative_bbox_area"].describe())

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Bounding Boxes Area per Class."

    @property
    def description(self) -> str:
        return (
            "The distribution of the areas of the boxes of the different classes as a histogram.\n"
            "The size of the objects can significantly affect the performance of your model. "
            "If certain classes tend to have smaller objects, the model might struggle to segment them, especially if the resolution of the images is low "
        )
