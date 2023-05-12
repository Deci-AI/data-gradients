from data_gradients.feature_extractors.features import ImageFeatures, SegmentationMaskFeatures
from data_gradients.feature_extractors.result import FeaturesCollection
from data_gradients.reports.report_interface import AbstractReportWidget
from data_gradients.visualize.plot_options import PlotRenderer, BarPlotOptions


class SegmentationClassDistribution(AbstractReportWidget):
    def __init__(self):
        pass

    def to_figure(self, results: FeaturesCollection, renderer: PlotRenderer):
        options = BarPlotOptions(
            x_label_key=SegmentationMaskFeatures.SegmentationMaskLabelName,
            x_label_name="Class name",
            y_label_key=None,
            y_label_name="Count",
            title="Class distribution",
            labels_key=ImageFeatures.DatasetSplit,
            labels_name="Split",
            x_ticks_rotation=45,
        )
        return renderer.render_with_options(results.mask_features, options)

    def to_json(self, results: FeaturesCollection):
        ...
