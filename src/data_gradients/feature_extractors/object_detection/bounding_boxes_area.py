import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.feature_extractors.utils import keep_most_frequent
from data_gradients.visualize.seaborn_renderer import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionBoundingBoxArea(AbstractFeatureExtractor):
    """Feature Extractor to compute the area covered Bounding Boxes."""

    def __init__(self, top_k: int = 8):
        self.data = []
        self.top_k = top_k
        self.n_classes = None

    def update(self, sample: DetectionSample):
        image_area = sample.image.shape[0] * sample.image.shape[1]
        for class_id, bbox_xyxy in zip(sample.class_ids, sample.bboxes_xyxy):
            class_name = sample.class_names[class_id]
            bbox_area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
            self.data.append(
                {
                    "split": sample.split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "relative_bbox_area": 100 * (bbox_area / image_area),
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)
        self.n_classes = len(df["class_name"].unique())

        plot_options = ViolinPlotOptions(
            x_label_key="relative_bbox_area",
            x_label_name="Bounding Box Area (in % of image)",
            y_label_key="class_name",
            y_label_name="Class",
            order_key="class_id",
            title=self.title,
            x_ticks_rotation=None,
            labels_key="split",
            bandwidth=0.4,
        )

        json = dict(
            train=dict(df[df["split"] == "train"]["relative_bbox_area"].describe()), val=dict(df[df["split"] == "val"]["relative_bbox_area"].describe())
        )

        df_to_plot = keep_most_frequent(df, filtering_key="class_name", frequency_key="relative_bbox_area", top_k=self.top_k)
        feature = Feature(
            data=df_to_plot,
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
            "The distribution of the areas of the boxes of the different classes.\n"
            "The size of the objects can significantly affect the performance of your model. "
            "If certain classes tend to have smaller objects, the model might struggle to segment them, especially if the resolution of the images is low "
        )

    @property
    def notice(self) -> str:
        if self.top_k is not None and self.n_classes is not None and self.n_classes > self.top_k:
            return (
                f"Only the <b>{self.top_k}/{self.n_classes}</b> most relevant features for this graph were shown.<br/>"
                f"You can increase/decrease the number of classes to plot by setting the parameter <b>`top_k`</b> in the configuration file."
            )
