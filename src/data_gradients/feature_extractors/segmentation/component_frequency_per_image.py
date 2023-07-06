import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.common import LABELS_PALETTE
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.plot_options import Hist2DPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationComponentsPerImageCount(AbstractFeatureExtractor):
    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):
        for j, class_channel in enumerate(sample.contours):
            self.data.append(
                {
                    "split": sample.split,
                    "sample_id": sample.sample_id,
                    "n_components": len(class_channel),
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = Hist2DPlotOptions(
            x_label_key="n_components",
            x_label_name="Number of component per Image",
            title=self.title,
            kde=False,
            labels_key="split",
            individual_plots_key="split",
            x_ticks_rotation=None,
            sharey=True,
            labels_palette=LABELS_PALETTE,
        )

        json = dict(
            train=dict(df[df["split"] == "train"]["n_components"].describe()),
            val=dict(df[df["split"] == "val"]["n_components"].describe()),
        )

        feature = Feature(data=df, plot_options=plot_options, json=json)
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Objects per Image"

    @property
    def description(self) -> str:
        return (
            "These graphs shows how many different objects appear in images. \n"
            "This can typically be valuable to know when you observe a very high number of objects per image, "
            "as some models include a parameter to filter the top k results."
        )
