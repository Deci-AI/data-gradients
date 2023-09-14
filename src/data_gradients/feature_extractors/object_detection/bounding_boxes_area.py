import math

import numpy as np
import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.seaborn_renderer import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.feature_extractors.utils import MostImportantValuesSelector


@register_feature_extractor()
class DetectionBoundingBoxArea(AbstractFeatureExtractor):
    """Feature Extractor to compute the area covered Bounding Boxes."""

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

        self.hist_transform_name = "sqrt"
        transforms = {"sqrt": lambda bbox_area: int(math.sqrt(bbox_area))}
        self.hist_transform = transforms[self.hist_transform_name]

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
                    f"bbox_area_{self.hist_transform_name}": self.hist_transform(bbox_area),
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        dict_bincount = self._compute_histogram(df=df, transform_name=self.hist_transform_name)

        df = self.value_extractor.select(df=df, id_col="class_id", split_col="split", value_col="relative_bbox_area")

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

        json = {}
        for split in df["split"].unique():
            basic_stats = dict(df[df["split"] == split]["relative_bbox_area"].describe())
            json[split] = {**basic_stats, "histogram_per_class": dict_bincount[split]}

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @staticmethod
    def _compute_histogram(df: pd.DataFrame, transform_name: str) -> dict:
        """
        Compute histograms for bounding box areas per class.

        :param df:                  DataFrame containing bounding box data.
        :param transform_name:      Type of transformation (like 'sqrt').
        :return:                    A dictionary containing relevant histogram information.
            Example:
            {
                'train': {
                    'transform': 'sqrt', # Transformation applied to the bbox area
                    'bin_width': 1,      # width between histogram bins. This depends on how the histogram is created.
                    'max_value': 3,      # max (transformed) area value included in the histogram
                    'histograms': {      # Dictionary of class name and its corresponding histogram
                        'A': [0, 1, 0, 2],
                        'B': [0, 0, 1, 0]
                    }
                },
                'val': ...
        }
        """
        max_value = df[f"bbox_area_{transform_name}"].max()
        max_value = int(max_value)

        dict_bincount = {}
        for split in df["split"].unique():
            dict_bincount[split] = {}
            split_data = df[df["split"] == split]

            dict_bincount[split] = {
                "transform": transform_name,
                "bin_width": 1,
                "max_value": max_value,
                "histograms": {},
            }

            for class_label in split_data["class_name"].unique():
                class_data = split_data[split_data["class_name"] == class_label]

                # Compute histograms for bin_width = 1
                bin_counts = np.bincount(class_data[f"bbox_area_{transform_name}"], minlength=max_value + 1)
                histogram = bin_counts.tolist()

                dict_bincount[split]["histograms"][class_label] = histogram

        return dict_bincount

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
