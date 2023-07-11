import collections
from typing import Optional

import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.feature_extractors.utils import MostImportantValuesSelector
from data_gradients.utils.data_classes.data_samples import ClassificationSample
from data_gradients.visualize.seaborn_renderer import BarPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class ClassificationClassFrequency(AbstractFeatureExtractor):
    """Feature Extractor to count the number of labels of each class."""

    def __init__(self, topk: Optional[int] = None, prioritization_mode: str = "train_val_diff"):
        """
        :param topk:                How many rows (per split) to show.
        :param prioritization_mode: Strategy to use to chose which class will be prioritized. Only the topk will be shown
                - 'train_val_diff': Returns the top k rows with the biggest train_val_diff between 'train' and 'val' split values.
                - 'outliers':       Returns the top k rows with the most extreme average values.
                - 'max':            Returns the top k rows with the highest average values.
                - 'min':            Returns the top k rows with the lowest average values.
                - 'min_max':        Returns the (top k)/2 rows with the biggest average values, and the (top k)/2 with the smallest average values.
        """
        if topk:
            self.value_extractor = MostImportantValuesSelector(topk=topk, prioritization_mode=prioritization_mode)
        else:
            self.value_extractor = None

        self.data = []

    def update(self, sample: ClassificationSample):
        class_name = sample.class_names[sample.class_id]
        self.data.append(
            {
                "split": sample.split,
                "class_id": sample.class_id,
                "class_name": class_name,
            }
        )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        # Include ("class_name", "split", "n_appearance")
        df_class_count = df.groupby(["class_name", "class_id", "split"]).size().reset_index(name="n_appearance")

        split_sums = df_class_count.groupby("split")["n_appearance"].sum()
        df_class_count["frequency"] = 100 * (df_class_count["n_appearance"] / df_class_count["split"].map(split_sums))

        all_class_names = df_class_count["class_name"].unique()

        if self.value_extractor:
            df_class_count = self.value_extractor.select(df=df_class_count, id_col="class_id", split_col="split", value_col="frequency")

        # Height of the plot is proportional to the number of classes
        n_unique = len(all_class_names)
        figsize_x = 10
        figsize_y = min(max(6, int(n_unique * 0.3)), 175)

        plot_options = BarPlotOptions(
            x_label_key="frequency",
            x_label_name="Frequency (%)",
            y_label_key="class_name",
            y_label_name="Class",
            order_key="class_id",
            title=self.title,
            figsize=(figsize_x, figsize_y),
            x_ticks_rotation=None,
            labels_key="split",
            orient="h",
            tight_layout=True,
        )

        json = df_class_count.to_json(orient="records")

        feature = Feature(
            data=df_class_count,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Class Frequency"

    @property
    def description(self) -> str:
        return (
            "This bar plot represents the frequency of appearance of each class. "
            "This may highlight class distribution gap between training and validation splits. "
            "For instance, if one of the class only appears in the validation set, you know in advance that your model won't be able to "
            "learn to predict that class."
        )
