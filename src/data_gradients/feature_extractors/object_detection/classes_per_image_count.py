import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.feature_extractors.utils import keep_most_frequent
from data_gradients.visualize.plot_options import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor


@register_feature_extractor()
class DetectionClassesPerImageCount(AbstractFeatureExtractor):
    """Feature Extractor to show the distribution of number of instance of each class per image.
    This gives information like "The class 'Human' usually appears 2 to 20 times per image."""

    def __init__(self, top_k: int = 30):
        self.data = []
        self.top_k = top_k
        self.n_classes = None

    def update(self, sample: DetectionSample):
        for class_id, bbox_xyxy in zip(sample.class_ids, sample.bboxes_xyxy):
            class_name = sample.class_names[class_id]
            self.data.append(
                {
                    "split": sample.split,
                    "sample_id": sample.sample_id,
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)
        self.n_classes = len(df["class_name"].unique())

        # Include ("split", "sample_id", "class_name", "class_id", "n_appearance")
        # For each split, image, class, I want to know how many bbox I have
        df_class_count = df.groupby(["split", "sample_id", "class_name", "class_id"]).size().reset_index(name="n_appearance")

        plot_options = ViolinPlotOptions(
            x_label_key="n_appearance",
            x_label_name="Number of class instance per Image",
            y_label_key="class_name",
            y_label_name="Class Names",
            order_key="class_id",
            title=self.title,
            bandwidth=0.4,
            x_ticks_rotation=None,
            labels_key="split",
        )

        json = dict(
            train=dict(df_class_count[df_class_count["split"] == "train"]["n_appearance"].describe()),
            val=dict(df_class_count[df_class_count["split"] == "val"]["n_appearance"].describe()),
        )

        df_to_plot = keep_most_frequent(df_class_count, filtering_key="class_name", frequency_key="n_appearance", top_k=self.top_k)
        feature = Feature(
            data=df_to_plot,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Number of classes per image."

    @property
    def description(self) -> str:
        return "The total number of bounding boxes for each class, across all images."

    @property
    def notice(self) -> str:
        if self.top_k is not None and self.n_classes is not None and self.n_classes > self.top_k:
            return (
                f"Only the <b>{self.top_k}/{self.n_classes}</b> most relevant features for this graph were shown.<br/>"
                f"You can increase/decrease the number of classes to plot by setting the parameter <b>`top_k`</b> in the configuration file."
            )
