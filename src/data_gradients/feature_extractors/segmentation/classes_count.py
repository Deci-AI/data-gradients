import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import BarPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationClassesCount(AbstractFeatureExtractor):
    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):
        for j, class_channel in enumerate(sample.contours):
            for contour in class_channel:
                class_id = contour.class_id
                class_name = sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        # Include ("class_name", "class_id", "split", "n_appearance")
        df_class_count = df.groupby(["class_name", "class_id", "split"]).size().reset_index(name="n_appearance")

        split_sums = df_class_count.groupby("split")["n_appearance"].sum()
        df_class_count["normalized_n_appearance"] = 100 * (df_class_count["n_appearance"] / df_class_count["split"].map(split_sums))

        plot_options = BarPlotOptions(
            x_label_key="normalized_n_appearance",
            x_label_name="Number of Appearance (in % of the split)",
            y_label_key="class_name",
            y_label_name="Class Names",
            order_key="class_id",
            title=self.title,
            x_ticks_rotation=None,
            labels_key="split",
            orient="h",
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
        return "Number of classes."

    @property
    def description(self) -> str:
        return (
            "The total number of connected components for each class, across all images. \n"
            "If the average number of components per image is too high, it might be due to image noise or the "
            "presence of many segmentation blobs."
        )
