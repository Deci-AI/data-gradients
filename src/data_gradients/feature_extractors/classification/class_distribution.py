import collections

import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes.data_samples import ClassificationSample
from data_gradients.visualize.seaborn_renderer import BarPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class ClassificationClassDistribution(AbstractFeatureExtractor):
    """Feature Extractor to count the number of labels of each class."""

    def __init__(self):
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
        all_class_names = df_class_count["class_name"].unique()

        # Height of the plot is proportional to the number of classes
        n_unique = len(all_class_names)
        figsize_x = 10
        figsize_y = min(max(6, int(n_unique * 0.3)), 175)

        plot_options = BarPlotOptions(
            x_label_key="n_appearance",
            x_label_name="Class support (# of samples)",
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

        json = {}
        for split in df["split"].unique():
            empty_dict = {class_name: 0 for class_name in all_class_names}
            counter = collections.Counter(empty_dict)
            counter.update(df[df["split"] == split]["class_name"])
            json[split] = dict(counter)

        feature = Feature(
            data=df_class_count,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Class distribution"

    @property
    def description(self) -> str:
        return (
            "Number of appearance of each class. This may highlight class distribution gap between training and validation splits. \n"
            "For instance, if one of the class only appears in the validation set, you know in advance that your model won't be able to "
            "learn to predict that class."
        )
