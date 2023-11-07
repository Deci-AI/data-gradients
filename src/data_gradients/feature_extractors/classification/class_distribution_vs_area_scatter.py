import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes.data_samples import ClassificationSample
from data_gradients.visualize.plot_options import ScatterPlotOptions


@register_feature_extractor()
class ClassificationClassDistributionVsAreaPlot(AbstractFeatureExtractor):
    """
    Visualizes the spread of image widths and heights within each class and across data splits.

    This feature extractor creates a scatter plot to graphically represent the diversity of image dimensions associated with each class label and split
    in the dataset.
    By visualizing this data, users can quickly assess whether certain classes or splits contain images that are consistently larger or smaller than others,
    potentially indicating a need for data preprocessing or augmentation strategies to ensure model robustness.

    Key Uses:

    - Identifying classes with notably different average image sizes that may influence model training.
    - Detecting splits in the dataset where image size distribution is uneven, prompting the need for more careful split strategies or
    tailored data augmentation.
    """

    def __init__(self):
        self.data = []

    def update(self, sample: ClassificationSample):
        class_name = sample.class_names[sample.class_id]
        self.data.append(
            {
                "split": sample.split,
                "class_id": sample.class_id,
                "class_name": class_name,
                "image_rows": sample.image.shape[0],
                "image_cols": sample.image.shape[1],
            }
        )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = ScatterPlotOptions(
            x_label_key="image_cols",
            x_label_name="Image width (px)",
            y_label_key="image_rows",
            y_label_name="Image height (px)",
            figsize=(10, 10),
            x_ticks_rotation=None,
            labels_key="class_name",
            style_key="split",
            # orient="h",
            tight_layout=True,
        )

        df_summary = (
            df[["split", "class_name", "image_rows", "image_cols"]]
            .groupby(["split", "class_name", "image_rows", "image_cols"])
            .size()
            .reset_index(name="counts")
        )

        json = df_summary.to_dict(orient="records")

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
            title="Image size distribution per class",
            description=(
                "Distribution of image size (mean value of image width & height) with respect to assigned image label and (when possible) a split.\n"
                "This may highlight issues when classes in train/val has different image resolution which may negatively affect the accuracy of the model.\n"
                "If you see a large difference in image size between classes and splits - you may need to adjust data collection process or training regime:\n"
                " - When splitting data into train/val/test - make sure that the image size distribution is similar between splits.\n"
                " - If size distribution overlap between splits to too big - "
                "you can address this (to some extent) by using more agressize values for zoom-in/zoo-out augmentation at training time.\n"
            ),
        )
        return feature
