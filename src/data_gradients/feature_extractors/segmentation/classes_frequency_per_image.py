import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.plot_options import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationClassesPerImageCount(AbstractFeatureExtractor):
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
            "This graph shows how many times each class appears in an image. "
            "It highlights whether each class has a constant number of appearances per image, "
            "or whether there is variability in the number of appearances from image to image."
        )
