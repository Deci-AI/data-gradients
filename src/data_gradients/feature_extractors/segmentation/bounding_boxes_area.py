import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import SegmentationSample
from data_gradients.visualize.seaborn_renderer import ViolinPlotOptions
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.feature_extractors.utils import MostImportantValuesSelector


@register_feature_extractor()
class SegmentationBoundingBoxArea(AbstractFeatureExtractor):
    """
    Semantic Segmentation task feature extractor -
    Get all Bounding Boxes areas and plot them as a percentage of the whole image.
    """

    def __init__(self, topk: int = 30, prioritization_mode: str = "train_val_diff"):
        """
        :param topk:                How many rows (per split) to show.
        :param prioritization_mode: Strategy to use to chose which class will be prioritized. Only the topk will be shown
                - 'train_val_diff': Returns the top k rows with the biggest train_val_diff between 'train' and 'val' split values.
                - 'outliers':       Returns the top k rows with the most extreme average values.
                - 'max':            Returns the top k rows with the highest average values.
                - 'min':            Returns the top k rows with the lowest average values.
                - 'min_max':        Returns the (top k)/2 rows with the biggest average values, and the (top k)/2 with the smallest average values.
        """
        self.value_extractor = MostImportantValuesSelector(topk=topk, prioritization_mode=prioritization_mode)
        self.data = []

    def update(self, sample: SegmentationSample):
        image_area = sample.image.shape[0] * sample.image.shape[1]
        for class_channel in sample.contours:
            for contour in class_channel:
                class_id = contour.class_id
                class_name = sample.class_names[class_id]
                self.data.append(
                    {
                        "split": sample.split,
                        "class_name": class_name,
                        "class_id": class_id,
                        "relative_bbox_area": 100 * (contour.bbox_area / image_area),
                    }
                )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data)

        df = self.value_extractor.select(df=df, id_col="class_id", split_col="split", value_col="relative_bbox_area")

        # Height of the plot is proportional to the number of classes
        n_unique = len(df["class_name"].unique())
        figsize_x = 10
        figsize_y = min(max(6, int(n_unique * 0.3)), 175)

        max_area = min(100, df["relative_bbox_area"].max())
        plot_options = ViolinPlotOptions(
            x_label_key="relative_bbox_area",
            x_label_name="Object Area (in % of image)",
            y_label_key="class_name",
            y_label_name="Class",
            order_key="class_id",
            title=self.title,
            figsize=(figsize_x, figsize_y),
            x_lim=(0, max_area),
            x_ticks_rotation=None,
            labels_key="split",
            bandwidth=0.4,
            tight_layout=True,
        )
        json = {split: dict(df[df["split"] == split]["relative_bbox_area"].describe()) for split in df["split"].unique()}

        feature = Feature(
            data=df,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Object Area"

    @property
    def description(self) -> str:
        return (
            "This graph shows the frequency of each class's appearance in the dataset. "
            "This can highlight distribution gap in object size between the training and validation splits, which can harm the model's performance. \n"
            "Another thing to keep in mind is that having too many very small objects may indicate that your are downsizing your original image to a "
            "low resolution that is not appropriate for your objects."
        )
