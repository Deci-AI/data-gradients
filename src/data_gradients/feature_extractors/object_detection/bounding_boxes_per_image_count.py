import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.common import LABELS_PALETTE
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.plot_options import Hist2DPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionBoundingBoxPerImageCount(AbstractFeatureExtractor):
    """Feature Extractor to count the number of Bounding Boxes per Image."""

    def __init__(self):
        self.data = []

    def update(self, sample: DetectionSample):
        for _ in sample.class_ids:
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
            x_label_name="Number of bounding box per Image",
            title=self.title,
            kde=False,
            labels_key="split",
            individual_plots_key="split",
            stat="percent",
            x_ticks_rotation=None,
            sharey=True,
            labels_palette=LABELS_PALETTE,
        )

        json = dict(
            train=dict(df_class_count[df_class_count["split"] == "train"]["n_components"].describe()),
            val=dict(df_class_count[df_class_count["split"] == "val"]["n_components"].describe()),
        )

        feature = Feature(data=df_class_count, plot_options=plot_options, json=json)
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Bounding Box per image"

    @property
    def description(self) -> str:
        return (
            "These graphs shows how many bounding boxes appear in images. \n"
            "This can typically be valuable to know when you observe a very high number of bounding boxes per image, "
            "as some models include a parameter to filter the top k results."
        )
