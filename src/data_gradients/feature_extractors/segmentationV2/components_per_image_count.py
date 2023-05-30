import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.plot_options import Hist2DPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationComponentsPerImageCount(AbstractFeatureExtractor):
    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):
        for j, class_channel in enumerate(sample.contours):
            for contour in class_channel:
                self.data.append(
                    {
                        "split": sample.split,
                        "sample_id": sample.sample_id,
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        # Include ("sample_id", "split", "n_components")
        df_class_count = df.groupby(["sample_id", "split"]).size().reset_index(name="n_components")

        plot_options = Hist2DPlotOptions(
            x_label_key="n_components",
            x_label_name="Number of component per Image",
            title=self.title,
            kde=True,
            labels_key="split",
            individual_plots_key="split",
            x_ticks_rotation=None,
        )

        json = dict(df_class_count.n_components.describe())

        feature = Feature(data=df_class_count, plot_options=plot_options, json=json)
        return feature

    @property
    def title(self) -> str:
        return "Number of component per image."

    @property
    def description(self) -> str:
        return "The total number of connected components per image. This helps understanding how many components images typically includes. "