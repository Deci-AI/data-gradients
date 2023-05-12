from data_gradients.feature_extractors.features import ImageFeatures
from data_gradients.feature_extractors.result import FeaturesResult
from data_gradients.reports.report_interface import AbstractReportWidget
from data_gradients.visualize.plot_options import PlotRenderer, Hist2DPlotOptions


class ImageSizeDistribution(AbstractReportWidget):
    def __init__(self):
        pass

    def to_figure(self, results: FeaturesResult, renderer: PlotRenderer):
        options = Hist2DPlotOptions(
            x_label_key=ImageFeatures.ImageWidth,
            x_label_name="Image width",
            y_label_key=ImageFeatures.ImageHeight,
            y_label_name="Image height",
            title="Image size distribution",
            labels_key=ImageFeatures.DatasetSplit,
            labels_name="Split",
        )
        return renderer.render_with_options(results.image_features, options)

    def to_json(self, results: FeaturesResult):
        return {
            "average_width": results.image_features[ImageFeatures.ImageWidth].mean(),
            "average_height": results.image_features[ImageFeatures.ImageHeight].mean(),
            "std_width": results.image_features[ImageFeatures.ImageWidth].std(),
            "std_height": results.image_features[ImageFeatures.ImageHeight].std(),
            "min_width": results.image_features[ImageFeatures.ImageWidth].min(),
            "max_width": results.image_features[ImageFeatures.ImageWidth].max(),
            "min_height": results.image_features[ImageFeatures.ImageHeight].min(),
            "max_height": results.image_features[ImageFeatures.ImageHeight].max(),
        }
