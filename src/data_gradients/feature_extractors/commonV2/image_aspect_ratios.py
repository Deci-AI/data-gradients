import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.visualize.plot_options import Hist2DPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature


@register_feature_extractor()
class ImagesAspectRatio(AbstractFeatureExtractor):
    """Extracts the distribution of the image aspect ratio."""

    def __init__(self):
        super().__init__()
        self.data = []

    def update(self, sample: ImageSample):
        height, width = sample.image.shape[:2]
        self.data.append({"split": sample.split, "height": height, "width": width})

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)
        title = "Image Aspect Ratios"

        plot_options = Hist2DPlotOptions(
            x_label_key="width",
            x_label_name="Width",
            y_label_key="height",
            y_label_name="Height",
            title=title,
            x_lim=(0, df["width"].max() + 100),
            y_lim=(0, df["height"].max() + 100),
            x_ticks_rotation=None,
            labels_key="split",
            individual_plots_key="split",
            individual_plots_max_cols=2,
        )
        description = df.describe()
        json = {"width": dict(description["width"]), "height": dict(description["height"])}

        feature = Feature(
            title=title,
            description=self.description,
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def description(self) -> str:
        return "The distribution of the aspect ratios of the images as a discrete histogram."
