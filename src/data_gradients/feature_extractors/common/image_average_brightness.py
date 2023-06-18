import cv2
import numpy as np
import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.visualize.plot_options import KDEPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import Feature


@register_feature_extractor()
class ImagesAverageBrightness(AbstractFeatureExtractor):
    """Extracts the distribution of the image 'brightness'."""

    def __init__(self):
        self.image_format = None
        self.data = []

    def update(self, sample: ImageSample):

        if self.image_format is None:
            self.image_format = sample.image_format
        else:
            if self.image_format != sample.image_format:
                raise RuntimeError(
                    f"Inconstancy in the image format. The image format of the sample {sample.sample_id} is not the same as the previous sample."
                )

        if self.image_format == ImageChannelFormat.RGB:
            brightness = np.mean(cv2.cvtColor(sample.image, cv2.COLOR_RGB2LAB)[0])
        elif self.image_format == ImageChannelFormat.BGR:
            brightness = np.mean(cv2.cvtColor(sample.image, cv2.COLOR_BGR2LAB)[0])
        elif self.image_format == ImageChannelFormat.GRAYSCALE:
            brightness = np.mean(sample.image)
        elif self.image_format == ImageChannelFormat.UNKNOWN:
            brightness = np.mean(sample.image)
        else:
            raise ValueError(f"Unknown image format {sample.image_format}")

        self.data.append({"split": sample.split, "brightness": brightness})

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

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
            "This graph shows the distribution of the image brightness of each dataset. \n"
            "This may for instance uncover differences between the training and validation sets, "
            "such as the presence of exclusively daytime images in the training set and nighttime images in the validation set."
        )
