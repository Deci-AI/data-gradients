import cv2
import numpy as np
import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.visualize.plot_options import Hist2DPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature


@register_feature_extractor()
class ImageBrightness(AbstractFeatureExtractor):
    """Extracts the distribution of the image 'brightness'."""

    def __init__(self):
        self.data = []

    def update(self, sample: ImageSample):
        if sample.image_format == ImageChannelFormat.RGB:
            brightness = np.mean(cv2.cvtColor(sample.image, cv2.COLOR_RGB2LAB)[0])
        elif sample.image_format == ImageChannelFormat.BGR:
            brightness = np.mean(cv2.cvtColor(sample.image, cv2.COLOR_BGR2LAB)[0])
        elif sample.image_format == ImageChannelFormat.GRAYSCALE:
            brightness = np.mean(sample.image)
        elif sample.image_format == ImageChannelFormat.UNKNOWN:
            brightness = np.mean(sample.image)
        else:
            raise ValueError(f"Unknown image format {sample.image_format}")

        self.data.append({"split": sample.split, "brightness": brightness})

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = Hist2DPlotOptions(
            x_label_key="brightness",
            x_label_name="Brightness",
            title=self.title,
            x_lim=(0, 255),
            x_ticks_rotation=None,
            labels_key="split",
            individual_plots_key="split",
            individual_plots_max_cols=2,
        )
        json = dict(df.brightness.describe())

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Image Brightness."

    @property
    def description(self) -> str:
        return (
            "The distribution of the image 'lightness' (as L channel pixel value distribution in CIELAB color "
            "space, as a discrete histogram (divided into 10 bins). \n"
            "Image brightness distribution can reveal differences between the train and validation set. I.e. if "
            "the train set contains only day images while the validation set contains night images. "
        )
