import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.common import LABELS_PALETTE
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import Hist2DPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationBoundingBoxResolution(AbstractFeatureExtractor):
    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):

        height, width = sample.image.shape[:2]
        for j, class_channel in enumerate(sample.contours):
            for contour in class_channel:
                class_id = contour.class_id
                class_name = sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "class_name": class_name,
                        "height": 100 * (contour.h / height),  # TODO: Decide to divide it by image height or not...
                        "width": 100 * (contour.w / width),
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = Hist2DPlotOptions(
            x_label_key="width",
            x_label_name="Width (in % of image)",
            y_label_key="height",
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
        train_json = {"width": dict(train_description["width"]), "height": dict(train_description["height"])}

        val_description = df[df["split"] == "val"].describe()
        val_json = {"width": dict(val_description["width"]), "height": dict(val_description["height"])}

        json = {"train": train_json, "val": val_json}

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Object Width and Height"

    @property
    def description(self) -> str:
        return (
            "These heat maps illustrate the distribution of objects width and height per class. \n"
            "Large variations in object size can affect the model's ability to accurately recognize objects."
        )
