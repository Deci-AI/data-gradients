import pandas as pd
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.visualize.plot_options import KDEPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import Feature


@register_feature_extractor()
class ImageColorDistribution(AbstractFeatureExtractor):
    """
    Analyzes and presents the color intensity distribution across image datasets.

    This feature assesses the distribution of color intensities in images and provides detailed visualizations for each
    color channel. It is designed to highlight differences and consistencies in color usage between training and
    validation datasets, which can be critical for adjusting image preprocessing parameters or for enhancing data augmentation techniques.
    """

    def __init__(self):
        self.image_channels = None
        self.colors = None
        self.palette = {"Red": "red", "Green": "green", "Blue": "blue", "Grayscale": "gray", "Luminance": "red", "A": "green", "B": "blue"}
        self.pixel_frequency_per_channel_per_split = {}
        for split in ["train", "val"]:
            self.pixel_frequency_per_channel_per_split[split] = np.zeros(shape=(3, 256), dtype=np.int64)

    def update(self, sample: ImageSample):
        if self.colors is None:
            self.colors = sample.image_channels.channel_names
            for channel_name in sample.image_channels.channel_names:
                if channel_name not in self.palette:
                    self.palette[channel_name] = "black"

        if self.image_channels is None:
            self.image_channels = sample.image_channels

        pixel_frequency_per_channel = self.pixel_frequency_per_channel_per_split.get(sample.split)

        # We need this more complex logic because we cannot directly accumulate the images (this would take too much memory)
        # so we need to iteratively count the frequency per split and per color
        for i, color in enumerate(self.colors):
            pixel_frequency_per_channel[i] += np.histogram(sample.image[:, :, i], bins=256)[0]

    def aggregate(self) -> Feature:
        data = [
            {"split": split, "Color": color, "pixel_value": pixel_value, "n": n}
            for split, pixel_frequency_per_channel in self.pixel_frequency_per_channel_per_split.items()
            for color, pixel_frequency in zip(self.colors, pixel_frequency_per_channel)
            for pixel_value, n in zip(range(256), pixel_frequency)
            # This check ensures that we don't plot empty histograms (E.g split is missing)
            if np.sum(self.pixel_frequency_per_channel_per_split[split]) > 0
        ]
        df = pd.DataFrame(data)

        plot_options = KDEPlotOptions(
            x_label_key="pixel_value",
            x_label_name="Color Intensity",
            weights="n",
            x_lim=(0, 255),
            x_ticks_rotation=None,
            labels_key="Color",
            individual_plots_key="split",
            common_norm=True,
            bw_adjust=0.4,
            labels_palette=self.palette,
            sharey=True,
        )
        df_train = df[df["split"] == "train"]
        train_json = {color: dict(df_train[df_train["Color"] == color]["n"].describe()) for color in self.colors}

        df_val = df[df["split"] == "val"]
        val_json = {color: dict(df_val[df_val["Color"] == color]["n"].describe()) for color in self.colors}

        json = {"train": train_json, "val": val_json}

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
            title="Color Distribution",
            description=(
                "Visualize the spread of color intensities with a frequency distribution for each channel, delineated from darkest (0) to brightest (255). "
                "By comparing these distributions between training and validation sets, you can identify any significant variations that might affect model "
                "performance. "
                "For instance, if one dataset shows a higher concentration of darker values, it could suggest a need for lighting correction in preprocessing."
            ),
        )
        return feature
