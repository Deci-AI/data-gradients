from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.visualize.plot_options import HeatmapOptions

DEFAULT_SIZES = [
    (64, 64),
    (96, 96),
    (128, 128),
    (160, 160),
    (192, 192),
    (224, 224),
    (256, 256),
    (320, 320),
    (384, 384),
    (448, 448),
    (512, 512),
    (640, 640),
    (768, 768),
    (896, 896),
    (1024, 1024),
]

DEFAULT_AREA_THRESHOLDS = [1, 4, 9, 16, 64]


@register_feature_extractor()
class DetectionResizeImpact(AbstractFeatureExtractor):
    """Analyze the impact of image resizing on the visibility of bounding boxes.
    The extractor evaluates the bounding box sizes upon varying the image resizing dimensions and records how many bounding boxes shrink below a certain size.
    """

    def __init__(self, resizing_sizes: Optional[List[Tuple[int, int]]] = None, area_thresholds: Optional[List[int]] = None, include_median_size: bool = True):
        """
        :param resizing_sizes:      List of tuples, where each tuple represents the dimensions (width, height) to which the image is resized.
                                If None, a default list of sizes is used.
        :param area_thresholds:     List of integers representing the minimum pixel areas (width*height) of bounding boxes to consider.
                                    If None, a default list of thresholds is used.
        :param include_median_size: If True, the median size of the bounding boxes is included in the analysis.
        """
        super().__init__()
        self.resizing_sizes = resizing_sizes or DEFAULT_SIZES
        self.area_thresholds = area_thresholds or DEFAULT_AREA_THRESHOLDS
        self.include_median_size = include_median_size
        self.data = []

    def update(self, sample: DetectionSample):
        height, width = sample.image.shape[0], sample.image.shape[1]
        for bbox_xyxy in sample.bboxes_xyxy:
            self.data.append(
                {
                    "split": sample.split,
                    "image_height": height,
                    "image_width": width,
                    "bboxes_height": bbox_xyxy[3] - bbox_xyxy[1],
                    "bboxes_width": bbox_xyxy[2] - bbox_xyxy[0],
                }
            )

    def aggregate(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)

        median_size = (int(df["image_width"].median()), int(df["image_height"].median()))
        resizing_size = set(self.resizing_sizes)
        resizing_size.add(median_size)
        resizing_size = sorted(resizing_size, key=lambda x: x[0] + x[1])

        splits = sorted(df["split"].unique())
        data: Dict[str, np.ndarray] = {}
        for split in splits:
            split_df = df[df["split"] == split]

            rescale = {threshold: dict() for threshold in self.area_thresholds}
            for (image_rescale_width, image_rescale_height) in resizing_size:
                name = f"{image_rescale_width}x{image_rescale_height}"

                rescaled_bbox_height = split_df["bboxes_height"] * image_rescale_height / split_df["image_height"]
                rescaled_bbox_width = split_df["bboxes_width"] * image_rescale_width / split_df["image_width"]
                area = rescaled_bbox_height * rescaled_bbox_width
                for threshold in self.area_thresholds:
                    rescale[threshold][name] = int((area < threshold).mean() * 100)

            data[split] = pd.DataFrame(rescale).to_numpy()

        # Height of the plot is proportional to the number of classes
        figsize_x = min(max(10, len(self.area_thresholds)), 25)
        figsize_y = min(max(6, int(len(resizing_size) * 0.3)), 175)

        resizing_size_names = [f"{width}x{height}" if (width, height) != median_size else f"Median - {width}x{height}" for (width, height) in resizing_size]
        plot_options = HeatmapOptions(
            xticklabels=[f"Area < {threshold}" for threshold in self.area_thresholds],
            yticklabels=resizing_size_names,
            x_label_name="Bounding Box Area (in px^2)",
            y_label_name="Image Size (width x height)",
            cbar=True,
            fmt="d",
            cmap="rocket_r",
            annot=True,
            square=True,
            figsize=(figsize_x, figsize_y),
            tight_layout=True,
            x_ticks_rotation=90,
            title=self.title,
        )

        feature = Feature(
            data=data,
            plot_options=plot_options,
            json={},
        )
        return feature

    @property
    def title(self) -> str:
        return "Distribution of Bounding Boxes smaller than a given Threshold"

    @property
    def description(self) -> str:
        return (
            "This visualization demonstrates the consequences of rescaling images on the visibility of their bounding boxes. <br/>"
            ""
            "By showcasing how bounding box sizes are affected upon varying the image resizing dimensions, we address a critical question: "
            '"<em>How far can we resize an image without causing its bounding boxes to shrink beyond a certain size, especially to less than 1px?</em>".<br/>'
            ""
            "Since an object, when scaled down to less than 1px, essentially disappears from the image, "
            "this analysis serves as a guide in identifying the optimal resizing limits that prevent crucial object data loss. <br/>"
            ""
            "Understanding this is crucial, as inappropriate resizing can result in significant object detail loss, "
            "thereby adversely affecting the performance of your model. "
        )
