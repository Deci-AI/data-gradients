import numpy as np

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstract import (
    FeatureExtractorAbstract,
)
from data_gradients.feature_extractors.utils import align_histogram_keys
from data_gradients.utils.data_classes import ImageSample
from data_gradients.utils.data_classes.extractor_results import HistogramResults


@register_feature_extractor()
class MeanAndSTD(FeatureExtractorAbstract):
    """
    Extracts the mean and std of the pixel values for each channel across all images (Blue-Mean, Blue-STD,
    Green-Mean, Green-STD, Red-Mean, Red-STD). Assumes BGR Channel ordering"
    """
    def __init__(self):
        super().__init__()
        self._hist = {"train": {"mean": [], "std": []}, "val": {"mean": [], "std": []}}

    def update(self, sample: ImageSample):
        self._hist[sample.split]["mean"].append(np.mean(sample.image, axis=(0, 1)))
        self._hist[sample.split]["std"].append(np.std(sample.image, axis=(0, 1)))

    def _aggregate(self, split: str):

        self._hist["train"], self._hist["val"] = align_histogram_keys(self._hist["train"], self._hist["val"])
        bgr_means = np.zeros(3)
        bgr_std = np.zeros(3)
        for channel in range(3):
            means = [self._hist[split]["mean"][i][channel].item() for i in range(len(self._hist[split]["mean"]))]
            bgr_means[channel] = np.mean(means)
            stds = [self._hist[split]["std"][i][channel].item() for i in range(len(self._hist[split]["std"]))]
            bgr_std[channel] = np.mean(stds)
        values = [bgr_means[0], bgr_std[0], bgr_means[1], bgr_std[1], bgr_means[2], bgr_std[2]]
        bins = ["Blue-Mean", "Blue-STD", "Green-Mean", "Green-STD", "Red-Mean", "Red-STD"]

        results = HistogramResults(
            bin_names=bins,
            bin_values=values,
            plot="bar-plot",
            split=split,
            color=self.colors[split],
            title="Images mean & std",
            y_label="Mean / STD",
            ticks_rotation=0,
            y_ticks=True,
        )
        return results

    @property
    def description(self) -> str:
        return "The mean and std of the pixel values for each channel across all images (Blue-Mean, Blue-STD, "\
              "Green-Mean, Green-STD, Red-Mean, Red-STD). Assumes BGR Channel ordering. \n" \
               "Can reveal " \
               "differences in the nature of the images in the two datasets or in the augmentation. I.e., if the mean " \
               "of one of the colors is shifted between the datasets, it might indicate wrong augmentation. "
