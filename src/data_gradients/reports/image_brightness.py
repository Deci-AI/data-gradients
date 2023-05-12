from data_gradients.feature_extractors.features import ImageFeatures
from data_gradients.feature_extractors.result import FeaturesResult
from data_gradients.visualize.plot_options import PlotRenderer, BarPlotOptions


class AverageImageBrightness:
    def __init__(self):
        pass

    def to_figure(self, results: FeaturesResult, renderer: PlotRenderer):
        options = BarPlotOptions(
            x_label_key=ImageFeatures.DatasetSplit,
            x_label_name="Split",
            y_label_key=ImageFeatures.ImageAvgBrightness,
            y_label_name="Average brightness",
            title="Average image brightness",
            log_scale=False,
            # labels_key=ImageFeatures.DatasetSplit,
            # labels_name="Split",
        )
        return renderer.render_with_options(results.image_features, options)

    def to_json(self, results: FeaturesResult):
        return {
            "average_brightness": results.image_features[ImageFeatures.ImageAvgBrightness].mean(),
        }
