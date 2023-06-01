import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.seaborn_renderer import Hist2DPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionBoundingBoxSize(AbstractFeatureExtractor):
    """Feature Extractor to gather the size (Height x Width) of Bounding Boxes."""

    def __init__(self):
        self.data = []

    def update(self, sample: DetectionSample):

        height, width = sample.image.shape[:2]
        for class_id, bbox_xyxy in zip(sample.class_ids, sample.bboxes_xyxy):
            class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]
            self.data.append(
                {
                    "split": sample.split,
                    "class_name": class_name,
                    "relative_height": 100 * ((bbox_xyxy[3] - bbox_xyxy[1]) / height),
                    "relative_width": 100 * ((bbox_xyxy[2] - bbox_xyxy[0]) / width),
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = Hist2DPlotOptions(
            x_label_key="relative_width",
            x_label_name="Width (in % of image)",
            y_label_key="relative_height",
            y_label_name="Height (in % of image)",
            title=self.title,
            x_lim=(0, 100),
            y_lim=(0, 100),
            x_ticks_rotation=None,
            labels_key="split",
            individual_plots_key="split",
            tight_layout=True,
        )

        description = df.describe()
        json = {"relative_width": dict(description["relative_width"]), "relative_height": dict(description["relative_height"])}

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Bounding Boxes Height and Width."

    @property
    def description(self) -> str:
        return (
            "Width, Height of the bounding-boxes surrounding every object across all images. Plotted per-class on a heat-map.\n"
            "A large variation in object sizes within a class can make it harder for the model to recognize the objects."
        )