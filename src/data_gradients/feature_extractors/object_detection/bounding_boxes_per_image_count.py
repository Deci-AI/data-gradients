import pandas as pd

from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.common import LABELS_PALETTE
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.plot_options import Hist2DPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.common.registry.registry import register_feature_extractor


@register_feature_extractor()
class DetectionBoundingBoxPerImageCount(AbstractFeatureExtractor):
    """
    Feature Extractor to count the number of Bounding Boxes per Image.

    It compiles the bounding box counts into a histogram distribution, allowing for easy identification
    of the frequency of bounding box occurrences across images in a dataset.
    """

    def __init__(self):
        self.data = []

    def update(self, sample: DetectionSample):
        self.data.append(
            {
                "split": sample.split,
                "sample_id": sample.sample_id,
                "n_bbox": len(sample.bboxes_xyxy),
            }
        )

    def aggregate(self) -> Feature:
        """
        Aggregate collected data into a histogram feature and provide descriptive statistics in JSON format.

        :return: An instance of Feature containing the aggregated data and plotting options.
        """
        df = pd.DataFrame(self.data)

        plot_options = Hist2DPlotOptions(
            x_label_key="n_bbox",
            x_label_name="Number of Bounding Boxes per Image",
            kde=False,
            labels_key="split",
            individual_plots_key="split",
            stat="percent",
            x_ticks_rotation=None,
            sharey=True,
            labels_palette=LABELS_PALETTE,
        )

        json = dict(
            train=dict(df[df["split"] == "train"]["n_bbox"].describe()),
            val=dict(df[df["split"] == "val"]["n_bbox"].describe()),
        )

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
            title="Distribution of Bounding Box per image",
            description=(
                "The histograms display the distribution of bounding box counts per image across dataset splits. "
                "They help to identify the commonality of bounding box frequencies, which can be instrumental "
                "in tuning detection models that process varying numbers of objects per image."
            ),
        )
        return feature
