import cv2
import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.visualize.plot_options import Hist2DPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature


@register_feature_extractor()
class ImageChannelsStats(AbstractFeatureExtractor):
    """Extracts the distribution of the image 'brightness'."""

    def __init__(self):
        self.data = []
        self.grayscale = False

    def update(self, sample: ImageSample):
        if sample.image_format == ImageChannelFormat.RGB:
            image = sample.image
        elif sample.image_format == ImageChannelFormat.BGR:
            image = cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)
        elif sample.image_format == ImageChannelFormat.GRAYSCALE:
            image = sample.image
            self.grayscale = True
        elif sample.image_format == ImageChannelFormat.UNKNOWN:
            image = sample.image
        else:
            raise ValueError(f"Unknown image format {sample.image_format}")

        # TODO: Find a way to do it lazy correctly...
        return image
        # self.data.append({"split": sample.split, "brightness": brightness})

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
        return "Distribution of Image Brightness"

    @property
    def description(self) -> str:
        # TODO: update
        return (
            "The mean and std of the pixel values for each channel across all images (Blue-Mean, Blue-STD, "
            "Green-Mean, Green-STD, Red-Mean, Red-STD). Assumes BGR Channel ordering. \n"
            "Can reveal "
            "differences in the nature of the images in the two datasets or in the augmentation. I.e., if the mean "
            "of one of the colors is shifted between the datasets, it might indicate wrong augmentation. "
        )
