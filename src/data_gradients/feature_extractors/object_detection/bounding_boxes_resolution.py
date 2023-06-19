import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.common import LABELS_PALETTE
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.seaborn_renderer import Hist2DPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionBoundingBoxSize(AbstractFeatureExtractor):
    """Feature Extractor to gather the size (Height x Width) of Bounding Boxes."""

    def __init__(self):
        self.data = []

    def update(self, sample: DetectionSample):

        height, width = sample.image.shape[:2]
        for class_id, bbox_xyxy in zip(sample.class_ids, sample.bboxes_xyxy):
            class_name = sample.class_names[class_id]
            self.data.append(
                {
                    "split": sample.split,
                    "class_name": class_name,
                    "relative_height": 100 * ((bbox_xyxy[3] - bbox_xyxy[1]) / height),
                    "relative_width": 100 * ((bbox_xyxy[2] - bbox_xyxy[0]) / width),
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = Hist2DPlotOptions(
            x_label_key="relative_width",
            x_label_name="Width (in % of image)",
            y_label_key="relative_height",
            y_label_name="Height (in % of image)",
            title=self.title,
            x_lim=(0, 100),
            y_lim=(0, 100),
            x_ticks_rotation=None,
            labels_key="split",
            individual_plots_key="split",
            tight_layout=False,
            sharey=True,
            labels_palette=LABELS_PALETTE,
        )

        train_description = df[df["split"] == "train"].describe()
        train_json = {"relative_width": dict(train_description["relative_width"]), "relative_height": dict(train_description["relative_height"])}

        val_description = df[df["split"] == "val"].describe()
        val_json = {"relative_width": dict(val_description["relative_width"]), "relative_height": dict(val_description["relative_height"])}

        json = {"train": train_json, "val": val_json}
        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Bounding Box Width and Height"

    @property
    def description(self) -> str:
        return (
            "These heat maps illustrate the distribution of bounding box width and height per class. \n"
            "Large variations in object size can affect the model's ability to accurately recognize objects."
        )
