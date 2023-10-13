import pandas as pd
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample
from data_gradients.visualize.plot_options import KDEPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import Feature


@register_feature_extractor()
class ImageColorDistribution(AbstractFeatureExtractor):
    """Extracts the distribution of the image 'brightness'."""

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
            title=self.title,
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
        )
        return feature

    @property
    def title(self) -> str:
        return "Color Distribution"

    @property
    def description(self) -> str:
        return (
            "Here's a comparison of image channel intensity (scaled 0-255) distributions across the entire dataset. \n"
            "It can reveal discrepancies in the image characteristics between the two datasets, as well as potential flaws in the augmentation process. \n"
            "E.g., a notable difference in the mean value of a specific color between the two datasets may indicate an issue with the augmentation process."
        )
