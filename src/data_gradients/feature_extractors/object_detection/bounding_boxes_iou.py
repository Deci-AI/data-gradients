import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes import DetectionSample
from data_gradients.visualize.plot_options import HeatmapOptions


@register_feature_extractor()
class DetectionBoundingBoxIoU(AbstractFeatureExtractor):
    """Feature Extractor to compute the pairwise IoU of bounding boxes per image.
    This feature extractor helps to identify duplicate/highly overlapping bounding boxes."""

    def __init__(self, num_bins: int, class_agnostic: bool = False):
        """
        :param num_bins: Number of bins to use for the heatmap plot.
        :param class_agnostic: If True, only check IoU of bounding boxes of the same class.
        """
        self.data = []
        self.num_bins = num_bins
        self.class_agnostic = class_agnostic

    def update(self, sample: DetectionSample):
        if len(sample.bboxes_xyxy) == 0:
            return

        bboxes = torch.from_numpy(sample.bboxes_xyxy)
        iou = box_iou(bboxes, bboxes).numpy()
        iou[np.eye(iou.shape[0], dtype=bool)] = 0

        ii, jj = np.nonzero(iou)
        for i, j in zip(ii, jj):
            self.data.append(
                {
                    "split": sample.split,
                    "class_id": sample.class_ids[i],
                    "class_name": sample.class_names[sample.class_ids[i]],
                    "other_class_id": sample.class_ids[j],
                    "other_class_name": sample.class_names[sample.class_ids[j]],
                    "iou": iou[i, j],
                }
            )

    def aggregate(self) -> Feature:
        df = pd.DataFrame(self.data).sort_values(by="class_id")

        num_classes = len(df["class_name"].unique())

        bins = np.linspace(0, 1, self.num_bins + 1)
        df["iou_bins"] = np.digitize(df["iou"].values, bins=bins)
        iou_bin_names = [f"[0..{bins[x]:.2f})" for x in range(1, len(bins))]
        iou_bin_names = [f"IoU < {bins[x]:.2f}" for x in range(1, len(bins))]

        class_names = list(df["class_name"].unique())
        splits = df["split"].unique()

        if not self.class_agnostic:
            df = df[df["class_id"] == df["other_class_id"]]

        data = {}
        json = {}

        for split in splits:
            counts = np.zeros((num_classes, self.num_bins), dtype=int)
            for i, class_name in enumerate(class_names):
                for j, iou_bin_name in enumerate(iou_bin_names):
                    counts[i, j] = len(df[(df["class_name"] == class_name) & (df["iou_bins"] == j + 1) & (df["split"] == split)])

            counts = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1].astype(np.float32)
            json[split] = counts.tolist()

            # Add "All classes" label
            counts = np.concatenate([counts, np.sum(counts, axis=0, keepdims=True)], axis=0)
            normalized_counts = (100 * counts / np.clip(counts[:, 0:1], a_min=1, a_max=None)).astype(int)

            data[split] = normalized_counts

        plot_options = HeatmapOptions(
            xticklabels=iou_bin_names,
            yticklabels=class_names + ["All classes"],
            x_label_name="IoU range",
            y_label_name="Class",
            cbar=True,
            fmt="d",
            cmap="rocket_r",
            annot=True,
            title=self.title,
            square=True,
            figsize=(10, (int(num_classes * 0.3) + 4) * len(splits)),
            tight_layout=True,
            x_ticks_rotation=90,
        )

        feature = Feature(
            data=data,
            plot_options=plot_options,
            json=json,
        )
        return feature

    @property
    def title(self) -> str:
        return "Intersection of bounding boxes"

    @property
    def description(self) -> str:
        description = (
            "The distribution of the box IoU with respect to other boxes in the sample. "
            "The heatmap shows the percentage of boxes with IoU in range [0..T] for each class. "
        )
        if self.class_agnostic:
            description += "Intersection of all boxes are considered (Regardless of classes of corresponding bboxes)."
        else:
            description += "Only intersection of boxes of same class are considered."
        return description
