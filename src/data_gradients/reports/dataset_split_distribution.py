from typing import Dict

from data_gradients.feature_extractors.features import ImageFeatures
from data_gradients.feature_extractors.result import FeaturesResult
from data_gradients.reports.report_interface import AbstractReportWidget
from data_gradients.visualize.plot_options import PlotRenderer, BarPlotOptions


class DatasetSplitDistribution(AbstractReportWidget):
    def __init__(self):
        pass

    def to_figure(self, results: FeaturesResult, renderer: PlotRenderer):
        options = BarPlotOptions(
            x_label_key=ImageFeatures.DatasetSplit,
            x_label_name="Split",
            y_label_key=None,
            y_label_name="Images",
            title="Image size distribution",
        )
        return renderer.render_with_options(results.image_features, options)

    def to_json(self, results: FeaturesResult) -> Dict[str, int]:
        images_per_split = results.image_features.groupby(ImageFeatures.DatasetSplit).count()
        return {split: images_per_split.loc[split].values[0] for split in images_per_split.index}
