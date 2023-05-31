import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.seaborn_renderer import BarPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionClassesCount(AbstractFeatureExtractor):
    """Feature Extractor to count the number of instance of each class."""

    def __init__(self):
        self.data = []

    def update(self, sample: DetectionSample):
        for class_id, bbox_xyxy in zip(sample.class_ids, sample.bboxes_xyxy):
            class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]
            self.data.append(
                {
                    "split": sample.split,
                    "class_name": class_name,
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        # Include ("class_name", "split", "n_appearance")
        df_class_count = df.groupby(["class_name", "split"]).size().reset_index(name="n_appearance")

        plot_options = BarPlotOptions(
            x_label_key="n_appearance",
            x_label_name="Number of Appearance",
            y_label_key="class_name",
            y_label_name="Class Names",
            title=self.title,
            x_ticks_rotation=None,
            labels_key="split",
            orient="h",
        )

        json = dict(df_class_count.class_name.describe())

        feature = Feature(
            data=df_class_count,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Number of classes."

    @property
    def description(self) -> str:
        return "The total number of bounding boxes for each class, across all images."
