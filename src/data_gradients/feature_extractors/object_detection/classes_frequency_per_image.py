import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.plot_options import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionClassesPerImageCount(AbstractFeatureExtractor):
    """Feature Extractor to show the distribution of number of instance of each class per image.
    This gives information like "The class 'Human' usually appears 2 to 20 times per image."""

    def __init__(self):
        self.data = []

    def update(self, sample: DetectionSample):
        for class_id, bbox_xyxy in zip(sample.class_ids, sample.bboxes_xyxy):
            class_name = sample.class_names[class_id]
            self.data.append(
                {
                    "split": sample.split,
                    "sample_id": sample.sample_id,
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        # Include ("class_name", "class_id", "split", "n_appearance")
        # For each class, image, split, I want to know how many bbox I have
        # TODO: check this
        df_class_count = df.groupby(["class_name", "class_id", "sample_id", "split"]).size().reset_index(name="n_appearance")

        plot_options = ViolinPlotOptions(
            x_label_key="n_appearance",
            x_label_name="Number of class instance per Image",
            y_label_key="class_name",
            y_label_name="Class Names",
            order_key="class_id",
            title=self.title,
            x_lim=(0, df_class_count["n_appearance"].max() * 1.2),
            bandwidth=0.4,
            x_ticks_rotation=None,
            labels_key="split",
        )

        json = dict(
            train=dict(df_class_count[df_class_count["split"] == "train"]["n_appearance"].describe()),
            val=dict(df_class_count[df_class_count["split"] == "val"]["n_appearance"].describe()),
        )

        feature = Feature(
            data=df_class_count,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Class Frequency per Image"

    @property
    def description(self) -> str:
        return (
            "This graph shows how many times each class appears in an image. It highlights whether each class has a constant number of "
            "appearance per image, or whether it really depends from an image to another."
        )
