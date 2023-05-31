import cv2
import pandas as pd
from typing import Dict
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.commonV2.utils import PixelFrequencyCounter
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.visualize.plot_options import KDEPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature


@register_feature_extractor()
class ImageColorDistribution(AbstractFeatureExtractor):
    """Extracts the distribution of the image 'brightness'."""

    def __init__(self):
        self.image_format = None
        self.colors = ("Red", "Green", "Blue")
        self.pixel_frequency_per_channel_per_split: Dict[str, Dict[str, PixelFrequencyCounter]] = {}
        self.palette = {"Red": "red", "Green": "green", "Blue": "blue", "Grayscale": "gray"}

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

        # We need this more complex logic because we cannot directly accumulate the images (this would take too much memory)
        # so we need to iteratively count the frequency per split and per color
        for i, color in enumerate(self.colors):
            pixel_frequency_per_channel = self.pixel_frequency_per_channel_per_split.get(sample.split, dict())
            pixel_frequency = pixel_frequency_per_channel.get(color, PixelFrequencyCounter())
            pixel_frequency.update(image[:, :, i])

            pixel_frequency_per_channel[color] = pixel_frequency
            self.pixel_frequency_per_channel_per_split[sample.split] = pixel_frequency_per_channel

    def aggregate(self) -> Feature:
        data = [
            {"split": split, "Color": color, "pixel_value": pixel_value, "count": count}
            for split, pixel_frequency_per_channel in self.pixel_frequency_per_channel_per_split.items()
            for color, pixel_frequency in pixel_frequency_per_channel.items()
            for pixel_value, count in pixel_frequency.compute().items()
        ]
        df = pd.DataFrame(data)

        plot_options = KDEPlotOptions(
            x_label_key="pixel_value",
            x_label_name="Color Intensity",
            weights="count",
            title=self.title,
            x_lim=(0, 255),
            x_ticks_rotation=None,
            labels_key="Color",
            individual_plots_key="split",
            individual_plots_max_cols=2,
            labels_palette=self.palette,
        )
        json = {color: dict(df[df["Color"] == color].describe()) for color in self.colors}

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Colors"

    @property
    def description(self) -> str:
        return (
            "Distribution of RBG or Grayscale intensity (0-255) over the whole dataset."
            "Assumes RGB Channel ordering: \n"
            "Can reveal differences in the nature of the images in the two datasets or in the augmentation. I.e., if the mean "
            "of one of the colors is shifted between the datasets, it might indicate wrong augmentation. "
        )
