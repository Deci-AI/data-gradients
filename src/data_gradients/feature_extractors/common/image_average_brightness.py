import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.visualize.plot_options import KDEPlotOptions
from data_gradients.visualize.plot_options import BarPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import Feature


@register_feature_extractor()
class ImagesAverageBrightness(AbstractFeatureExtractor):
    """Extracts the distribution of the image 'brightness'."""

    def __init__(self):
        self.image_channels = None
        self.data = []

    def update(self, sample: ImageSample):
        self.data.append({"split": sample.split, "brightness": sample.image_mean_intensity})

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)
        n_unique_per_split = {len(df[df["split"] == split]["brightness"].unique()) for split in df["split"].unique()}

        # If a split has only one unique value, KDE plot will not work. Instead, we show the average brightness of the images.
        if 1 in n_unique_per_split:
            plot_options = BarPlotOptions(
                x_label_key="split",
                x_label_name="Split",
                y_label_key="brightness",
                y_label_name="Average Brightness",
                title=self.title,
                x_ticks_rotation=None,
                orient="v",
                show_values=False,
            )
        else:
            plot_options = KDEPlotOptions(
                x_label_key="brightness",
                x_label_name="Average Brightness of Images",
                title=self.title,
                x_lim=(0, 255),
                x_ticks_rotation=None,
                labels_key="split",
                common_norm=False,
                fill=True,
                sharey=True,
            )

        json = dict(train=dict(df[df["split"] == "train"]["brightness"].describe()), val=dict(df[df["split"] == "val"]["brightness"].describe()))

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Image Brightness Distribution"

    @property
    def description(self) -> str:
        return (
            "This graph shows the distribution of the brightness levels across all images. \n"
            "This may for instance uncover differences between the training and validation sets, "
            "such as the presence of exclusively daytime images in the training set and nighttime images in the validation set."
        )
