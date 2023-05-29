import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.feature_extractor_abstractV2 import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.plot_options import ViolinPlotOptions
from data_gradients.feature_extractors.feature_extractor_abstractV2 import AbstractFeatureExtractor


@register_feature_extractor()
class SegmentationClassesPerImageCount(AbstractFeatureExtractor):
    def __init__(self):
        self.data = []

    def update(self, sample: SegmentationSample):
        for j, class_channel in enumerate(sample.contours):
            for contour in class_channel:
                class_id = contour.class_id
                class_name = str(class_id) if sample.class_names is None else sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "sample_id": sample.sample_id,
                        "class_name": class_name,
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        # Include ("class_name", "split", "n_appearance")
        # For each class, image, split, I want to know how many bbox I have
        # TODO: check this
        df_class_count = df.groupby(["class_name", "sample_id", "split"]).size().reset_index(name="n_appearance")

        plot_options = ViolinPlotOptions(
            x_label_key="n_appearance",
            x_label_name="Number of class instance per Image",
            y_label_key="class_name",
            y_label_name="Class Names",
            title=self.title,
            bandwidth=0.4,
            x_ticks_rotation=None,
            labels_key="split",
        )

        json = dict(df_class_count.n_appearance.describe())

        feature = Feature(
            data=df_class_count,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Number of classes per image."

    @property
    def description(self) -> str:
        return (
            "The total number of connected components for each class, across all images. \n"
            "If the average number of components per image is too high, it might be due to image noise or the "
            "presence of many segmentation blobs."
        )
