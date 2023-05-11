from data_gradients.feature_extractors.features import ImageFeatures, SegmentationMaskFeatures
from data_gradients.feature_extractors.result import FeaturesResult
from data_gradients.reports.report_interface import AbstractReportWidget
from data_gradients.visualize.plot_renderer import PlotRenderer, Hist2DPlotOptions, ScatterPlotOptions


class SegmentationMasksArea(AbstractReportWidget):
    def __init__(self):
        pass

    def to_figure(self, results: FeaturesResult, renderer: PlotRenderer):
        options = ScatterPlotOptions(
            x_label_key=SegmentationMaskFeatures.SegmentationMaskLabelName,
            x_label_name="Class name",
            y_label_key=SegmentationMaskFeatures.SegmentationMaskArea,
            y_label_name="Instance area",
            title="Segmentation masks area distribution",
            labels_key=ImageFeatures.DatasetSplit,
            labels_name="Split",
            # x_ticks_rotation=45,
            # bins=16,
        )
        return renderer.render_with_options(results.mask_features, options)

    def to_json(self, results: FeaturesResult):
        ...
