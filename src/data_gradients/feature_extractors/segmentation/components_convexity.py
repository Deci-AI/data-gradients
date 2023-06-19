import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import KDEPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.batch_processors.preprocessors import contours


@register_feature_extractor()
class SegmentationComponentsConvexity(AbstractFeatureExtractor):
    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):
        for j, class_channel in enumerate(sample.contours):
            for contour in class_channel:
                convex_hull = contours.get_convex_hull(contour)
                convex_hull_perimeter = contours.get_contour_perimeter(convex_hull)
                convexity_measure = (contour.perimeter - convex_hull_perimeter) / contour.perimeter
                self.data.append(
                    {
                        "split": sample.split,
                        "convexity_measure": convexity_measure,
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        plot_options = KDEPlotOptions(
            x_label_key="convexity_measure",
            x_label_name="Convexity",
            title=self.title,
            x_ticks_rotation=None,
            labels_key="split",
            common_norm=False,
            fill=True,
            sharey=True,
        )

        json = dict(train=dict(df[df["split"] == "train"]["convexity_measure"].describe()), val=dict(df[df["split"] == "val"]["convexity_measure"].describe()))

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Objects Convexity"

    @property
    def description(self) -> str:
        return (
            "This graph depicts the convexity distribution of objects in both training and validation sets. \n"
            "Higher convexity values suggest complex structures that may pose challenges for accurate segmentation."
        )
