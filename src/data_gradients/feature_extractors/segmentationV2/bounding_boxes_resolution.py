import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import Hist2DPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class BoundingBoxResolution(AbstractFeatureExtractor):
    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):

        height, width = sample.image.shape[:2]
        for j, class_channel in enumerate(sample.contours):
            for contour in class_channel:
                class_id = contour.class_id
                class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "class_name": class_name,
                        "height": 100 * (contour.h / height),  # TODO: Decide to divide it by image height or not...
                        "width": 100 * (contour.w / width),
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)
        title = "Distribution of Bounding Boxes Height and Width per Class"

        plot_options = Hist2DPlotOptions(
            x_label_key="width",
            x_label_name="Width (in % of image)",
            y_label_key="height",
            y_label_name="Height (in % of image)",
            title=title,
            x_lim=(0, 100),
            y_lim=(0, 100),
            x_ticks_rotation=None,
            labels_key="split",
            individual_plots_key="class_name",
            individual_plots_max_cols=3,
        )

        description = df.describe()
        json = {"width": dict(description["width"]), "height": dict(description["height"])}

        feature = Feature(
            title=title,
            description=self.description,
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def description(self) -> str:
        return (
            "Width, Height of the bounding-boxes surrounding every object across all images. Plotted per-class on a heat-map.\n"
            "A large variation in object sizes within a class can make it harder for the model to recognize the objects."
        )
