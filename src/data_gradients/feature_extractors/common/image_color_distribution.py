import cv2
import pandas as pd
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.visualize.plot_options import KDEPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import Feature


@register_feature_extractor()
class ImageColorDistribution(AbstractFeatureExtractor):
    """Extracts the distribution of the image 'brightness'."""

    def __init__(self):
        self.image_format = None
        self.colors = ("Red", "Green", "Blue")
        self.palette = {"Red": "red", "Green": "green", "Blue": "blue", "Grayscale": "gray"}
        self.pixel_frequency_per_channel_per_split = {}
        for split in ["train", "val"]:
            self.pixel_frequency_per_channel_per_split[split] = np.zeros(shape=(3, 256), dtype=np.int64)

    def update(self, sample: ImageSample):

        if self.image_format is None:
            self.image_format = sample.image_format
        else:
            if self.image_format != sample.image_format:
                raise RuntimeError(
                    f"Inconstancy in the image format. The image format of the sample {sample.sample_id} is not the same as the previous sample."
                )

        if self.image_format == ImageChannelFormat.RGB:
            image = sample.image
        elif self.image_format == ImageChannelFormat.BGR:
            image = cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)
        elif self.image_format == ImageChannelFormat.GRAYSCALE:
            image = sample.image[:, :, np.newaxis]
            self.colors = ("Grayscale",)
        elif self.image_format == ImageChannelFormat.UNKNOWN:
            image = sample.image
        else:
            raise ValueError(f"Unknown image format {sample.image_format}")

        sample.image = sample.image.astype(np.uint8)
        pixel_frequency_per_channel = self.pixel_frequency_per_channel_per_split.get(sample.split)

        # We need this more complex logic because we cannot directly accumulate the images (this would take too much memory)
        # so we need to iteratively count the frequency per split and per color
        for i, color in enumerate(self.colors):
            pixel_frequency_per_channel[i] += np.histogram(image[:, :, i], bins=256)[0]

    def aggregate(self) -> Feature:
        data = [
            {"split": split, "Color": color, "pixel_value": pixel_value, "n": n}
            for split, pixel_frequency_per_channel in self.pixel_frequency_per_channel_per_split.items()
            for color, pixel_frequency in zip(self.colors, pixel_frequency_per_channel)
            for pixel_value, n in zip(range(256), pixel_frequency)
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
            "Here's a comparison of RGB or grayscale intensity intensity (0-255) distributions across the entire dataset, assuming RGB channel ordering. \n"
            "It can reveal discrepancies in the image characteristics between the two datasets, as well as potential flaws in the augmentation process. \n"
            "E.g., a notable difference in the mean value of a specific color between the two datasets may indicate an issue with augmentation."
        )
