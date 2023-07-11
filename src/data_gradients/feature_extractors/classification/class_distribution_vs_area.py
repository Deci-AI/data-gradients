import collections
from functools import partial

import numpy as np
import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes.data_samples import ClassificationSample
from data_gradients.visualize.plot_options import ViolinPlotOptions
from data_gradients.visualize.seaborn_renderer import BarPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class ClassificationClassDistributionVsArea(AbstractFeatureExtractor):
    """Feature Extractor to count the number of labels of each class."""

    def __init__(self):
        self.data = []

    def update(self, sample: ClassificationSample):
        class_name = sample.class_names[sample.class_id]
        self.data.append({"split": sample.split, "class_id": sample.class_id, "class_name": class_name, "image_area": np.prod(sample.image.shape[:2])})

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        all_class_names = df["class_name"].unique()

        num_splits = len(df["split"].unique())
        # Height of the plot is proportional to the number of classes
        n_unique = len(all_class_names)
        figsize_x = 10
        figsize_y = min(max(6, int(n_unique * 0.3)), 175)

        plot_options = ViolinPlotOptions(
            x_label_key="image_area",
            x_label_name="Image area (px²)",
            y_label_key="class_name",
            y_label_name="Class",
            order_key="class_id",
            title=self.title,
            figsize=(figsize_x, figsize_y),
            # x_lim=(0, df_class_count["n_appearance"].max() * 1.2),
            x_ticks_rotation=None,
            labels_key="split" if num_splits > 1 else None,
            # orient="h",
            tight_layout=True,
        )

        json = {}
        for split in df["split"].unique():
            empty_dict = {class_name: 0 for class_name in all_class_names}
            counter = collections.Counter(empty_dict)
            counter.update(df[df["split"] == split]["class_name"])
            json[split] = dict(counter)

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Image area distribution per class"

    @property
    def description(self) -> str:
        return (
            "Distribution of images resolution (H*W) with respect to assigned image label and (when possible) a split.\n"
            "This may highlight issues when classes in train/val has different image resolution which may negatively affect the accuracy of the model.\n"
        )
