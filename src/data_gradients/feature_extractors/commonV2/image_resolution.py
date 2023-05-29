import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.visualize.plot_options import Hist2DPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature


@register_feature_extractor()
class ImagesResolution(AbstractFeatureExtractor):
    """Extracts the distribution image Height and Width."""

    def __init__(self):
        super().__init__()
        self.data = []

    def update(self, sample: ImageSample):
        height, width = sample.image.shape[:2]
        self.data.append({"split": sample.split, "height": height, "width": width})

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = Hist2DPlotOptions(
            x_label_key="width",
            x_label_name="Width",
            y_label_key="height",
            y_label_name="Height",
            title=self.title,
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
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Extracts the distribution image Height and Width."

    @property
    def description(self) -> str:
        return (
            "The distribution of the image resolutions as a discrete histogram. \n Note that if images are "
            "rescaled or padded, this plot will show the size after rescaling and padding. "
        )
