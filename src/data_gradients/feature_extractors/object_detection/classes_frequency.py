import pandas as pd
from abc import ABC, abstractmethod
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.seaborn_renderer import BarPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.feature_extractors.utils import MostImportantValuesSelector


@register_feature_extractor()
class DetectionClassFrequency(AbstractFeatureExtractor):
    """Feature Extractor to count the number of instance of each class."""

    def __init__(self, topk: int = 40, mode: str = "gap"):
        self.value_extractor = MostImportantValuesSelector(topk=topk, mode=mode)
        self.data = []

    def update(self, sample: DetectionSample):
        for class_id, bbox_xyxy in zip(sample.class_ids, sample.bboxes_xyxy):
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

        # Include ("class_name", "split", "n_appearance")
        df_class_count = df.groupby(["class_name", "class_id", "split"]).size().reset_index(name="n_appearance")

        split_sums = df_class_count.groupby("split")["n_appearance"].sum()
        df_class_count["frequency"] = 100 * (df_class_count["n_appearance"] / df_class_count["split"].map(split_sums))

        df_class_count = self.value_extractor.select(df=df_class_count, id_col="class_id", split_col="split", value_col="frequency")

        # Height of the plot is proportional to the number of classes
        n_unique = len(df_class_count["class_name"].unique())
        figsize_x = 10
        figsize_y = min(max(6, int(n_unique * 0.3)), 175)

        plot_options = BarPlotOptions(
            x_label_key="frequency",
            x_label_name="Frequency",
            y_label_key="class_name",
            y_label_name="Class",
            order_key="class_id",
            title=self.title,
            figsize=(figsize_x, figsize_y),
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
        return "Class Frequency"

    @property
    def description(self) -> str:
        return (
            "Frequency of appearance of each class. This may highlight class distribution gap between training and validation splits. \n"
            "For instance, if one of the class only appears in the validation set, you know in advance that your model won't be able to "
            "learn to predict that class."
        )


class DataframeExtractor(ABC):
    def __init__(self, topk: int):
        self.topk = topk

    @abstractmethod
    def extract(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass


class OutliersExtractor(DataframeExtractor):
    def extract(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        values = df[column]
        values_normalized = (values - values.mean()) / values.var()
        outliers_index = values_normalized.abs().sort_values(ascending=False).index[: self.topk]
        return df[outliers_index]


class HighestValuesExtractor(DataframeExtractor):
    def extract(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        return df.sort_values(by=column, ascending=False)[: self.topk]


class LowestValuesExtractor(DataframeExtractor):
    def extract(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        return df.sort_values(by=column, ascending=True)[: self.topk]


def get_dataframe_extractor_per_frequency(extractor_name: str, topk: int) -> DataframeExtractor:
    available_extractors = {
        "outliers": OutliersExtractor(topk=topk),
        "most_frequent": HighestValuesExtractor(topk=topk),
        "least_frequent": LowestValuesExtractor(topk=topk),
    }
    if extractor_name not in available_extractors.keys():
        raise ValueError
    return available_extractors[extractor_name]
