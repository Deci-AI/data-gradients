from collections import defaultdict
from typing import List

import cv2
import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.utils.data_classes.data_samples import ImageSample, ImageChannelFormat
from data_gradients.utils.data_classes.extractor_results import HistogramResults


@register_feature_extractor()
class AverageBrightness(FeatureExtractorAbstract):
    """
    Average brightness feature extractor.
    Extracts the distribution of the image 'lightness' (as L channel pixel value distribution in CIELAB
    color space, as a discrete histogram (divided into 10 bins).

    """
    def __init__(self):
        super().__init__()
        self._num_bins: int = 10
        self._brightness_per_split = defaultdict(list)

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

        self._brightness_per_split[sample.split].append(brightness)

    def _aggregate(self, split: str) -> HistogramResults:
        values, bins = np.histogram(self._brightness_per_split[split], bins=self._num_bins)
        values = [np.round(((100 * value) / sum(list(values))), 3) for value in values]
        bins = self._create_keys(bins)
        results = HistogramResults(
            bin_names=bins,
            bin_values=list(values),
            plot="bar-plot",
            split=split,
            title="Average brightness of images",
            color=self.colors[split],
            y_label="% out of all images",
            y_ticks=True,
        )
        return results

    @staticmethod
    def _create_keys(bins):
        new_keys: List[str] = []
        for i, key in enumerate(bins):
            if i == len(bins) - 1:
                continue
            new_keys.append("{:.2f}<{:.2f}".format(bins[i], bins[i + 1]))

        return new_keys

    @property
    def description(self):
        return "The distribution of the image 'lightness' (as L channel pixel value distribution in CIELAB color " \
               "space, as a discrete histogram (divided into 10 bins). \n" \
               "Image brightness distribution can reveal differences between the train and validation set. I.e. if " \
               "the train set contains only day images while the validation set contains night images. "

