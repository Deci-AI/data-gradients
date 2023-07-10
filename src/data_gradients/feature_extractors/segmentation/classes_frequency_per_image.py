import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.plot_options import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.feature_extractors.utils import MostImportantValuesSelector


@register_feature_extractor()
class SegmentationClassesPerImageCount(AbstractFeatureExtractor):
    def __init__(self, topk: int = 30, prioritization_mode: str = "train_val_diff"):
        """
        :param topk:                How many rows (per split) to show.
        :param prioritization_mode: Strategy to use to chose which class will be prioritized. Only the topk will be shown
                - 'train_val_diff': Returns the top k rows with the biggest train_val_diff between 'train' and 'val' split values.
                - 'outliers':       Returns the top k rows with the most extreme average values.
                - 'max':            Returns the top k rows with the highest average values.
                - 'min':            Returns the top k rows with the lowest average values.
                - 'min_max':        Returns the (top k)/2 rows with the biggest average values, and the (top k)/2 with the smallest average values.
        """
        self.value_extractor = MostImportantValuesSelector(topk=topk, prioritization_mode=prioritization_mode)
        self.data = []

    def update(self, sample: SegmentationSample):

        for j, class_channel in enumerate(sample.contours):
            for contour in class_channel:
                class_id = contour.class_id
                class_name = sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "sample_id": sample.sample_id,
                        "class_name": class_name,
                        "class_id": class_id,
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        # Include ("class_name", "class_id", "split", "n_appearance")
        # For each class, image, split, I want to know how many bbox I have
        df_class_count = df.groupby(["class_name", "class_id", "sample_id", "split"]).size().reset_index(name="n_appearance")

        df_class_count = self.value_extractor.select(df=df_class_count, id_col="class_id", split_col="split", value_col="n_appearance")

        max_n_appearance = df_class_count["n_appearance"].max()

        # Height of the plot is proportional to the number of classes
        n_unique = len(df_class_count["class_name"].unique())
        figsize_x = 10
        figsize_y = min(max(6, int(n_unique * 0.3)), 175)

        plot_options = ViolinPlotOptions(
            x_label_key="n_appearance",
            x_label_name="Number of class instance per Image",
            y_label_key="class_name",
            y_label_name="Class Names",
            order_key="class_id",
            title=self.title,
            x_lim=(0, max_n_appearance * 1.2),  # Cut the max_x at 120% of the highest max n_appearance to increase readability
            figsize=(figsize_x, figsize_y),
            bandwidth=0.4,
            x_ticks_rotation=None,
            labels_key="split",
            tight_layout=True,
        )

        json = {split: dict(df_class_count[df_class_count["split"] == split]["n_appearance"].describe()) for split in df_class_count["split"].unique()}

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
            "This graph shows how many times each class appears in an image. "
            "It highlights whether each class has a constant number of appearances per image, "
            "or whether there is variability in the number of appearances from image to image."
        )
