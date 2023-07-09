import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.seaborn_renderer import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionBoundingBoxArea(AbstractFeatureExtractor):
    """Feature Extractor to compute the area covered Bounding Boxes."""

    def __init__(self):
        self.data = []

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

        # Height of the plot is proportional to the number of classes
        n_unique = len(df["class_name"].unique())
        figsize_x = 10
        figsize_y = min(max(6, int(n_unique * 0.3)), 175)

        max_area = min(100, df["relative_bbox_area"].max())

        plot_options = ViolinPlotOptions(
            x_label_key="relative_bbox_area",
            x_label_name="Bounding Box Area (in % of image)",
            y_label_key="class_name",
            y_label_name="Class",
            order_key="class_id",
            title=self.title,
            x_ticks_rotation=None,
            labels_key="split",
            x_lim=(0, max_area),
            figsize=(figsize_x, figsize_y),
            bandwidth=0.4,
            tight_layout=True,
        )

        json = dict(
            train=dict(df[df["split"] == "train"]["relative_bbox_area"].describe()), val=dict(df[df["split"] == "val"]["relative_bbox_area"].describe())
        )

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Bounding Box Area"

    @property
    def description(self) -> str:
        return (
            "This graph shows the frequency of each class's appearance in the dataset. "
            "This can highlight distribution gap in object size between the training and validation splits, which can harm the model's performance. \n"
            "Another thing to keep in mind is that having too many very small objects may indicate that your are downsizing your original image to a "
            "low resolution that is not appropriate for your objects."
        )
