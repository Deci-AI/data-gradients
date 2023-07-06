import dataclasses
import os
from typing import List

import numpy as np
from imagededup.methods import DHash
from jinja2 import Template

from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors import AbstractFeatureExtractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.assets import assets
from data_gradients.utils.data_classes.data_samples import ImageSample, SegmentationSample, DetectionSample


@dataclasses.dataclass
class BasicStatistics:

    image_count: int = 0
    classes_count: int = 0
    classes_in_use: int = 0
    classes: List[int] = dataclasses.field(default_factory=list)
    annotation_count: int = 0
    images_without_annotation: int = 0
    images_resolutions: List[int] = dataclasses.field(default_factory=list)
    annotations_sizes: List[int] = dataclasses.field(default_factory=list)
    annotations_per_image: List[int] = dataclasses.field(default_factory=list)
    med_image_resolution: int = 0
    smallest_annotations: int = 0
    largest_annotations: int = 0
    most_annotations: int = 0
    least_annotations: int = 0
    duplicate_images_in_train: int = 0
    duplicate_images_in_validation: int = 0
    duplicate_image_appearences_in_train: int = 0
    duplicate_image_appearences_in_validation: int = 0


@register_feature_extractor()
class SummaryStats(AbstractFeatureExtractor):
    """Extracts general summary statistics from images."""

    def __init__(self, find_duplicates: bool = True):
        super().__init__()
        self.stats = {"train": BasicStatistics(), "val": BasicStatistics()}
        self.find_duplicates = find_duplicates
        self.template = Template(source=assets.html.basic_info_fe)
        self._train_image_dir = None
        self._valid_image_dir = None

    def _get_image_dir(self, split: str) -> str:
        p = input(f"Image duplicates extraction: please enter the full path to the directory containing all {split} images >>> \n")
        if not os.path.exists(p):
            raise ValueError(f"Path to the directory containing all {split} images does not exist.")
        return p

    def _is_in_dup_clique(self, sample, dup_clique_list):
        return any([sample in d for d in dup_clique_list])

    def _make_dup_clique(self, dup_key, dups):
        dup_clique = [dup_key] + dups[dup_key]
        return dup_clique

    def _is_train_dup(self, dup_clique):
        return len([d for d in dup_clique if d.startswith(self._train_image_dir)]) > 1

    def _is_valid_dup(self, dup_clique):
        if self._valid_image_dir is None:
            return False
        else:
            return len([d for d in dup_clique if d.startswith(self._valid_image_dir)]) > 1

    def _is_intersection_dup(self, dup_clique):
        return (
            len([d for d in dup_clique if d.startswith(self._train_image_dir)]) > 0 and len([d for d in dup_clique if d.startswith(self._valid_image_dir)]) > 0
        )

    def update(self, sample: ImageSample):

        basic_stats = self.stats[sample.split]

        height, width = sample.image.shape[:2]
        basic_stats.images_resolutions.append([height, width])

        basic_stats.image_count += 1

        if isinstance(sample, SegmentationSample):
            contours = [contour for sublist in sample.contours for contour in sublist]
            basic_stats.annotations_per_image.append(len(contours))

            for contour in contours:
                basic_stats.annotations_sizes.append(contour.area)
                basic_stats.classes.append(contour.class_id)

            basic_stats.classes_count = len(sample.class_names)

        elif isinstance(sample, DetectionSample):
            labels = sample.class_ids
            basic_stats.classes.extend(labels)
            boxes = sample.bboxes_xyxy
            basic_stats.annotations_per_image.append(len(boxes))
            for box in boxes:
                basic_stats.annotations_sizes.append((box[2] - box[0]) * (box[3] - box[1]))

            basic_stats.classes_count = len(sample.class_names)

    def aggregate(self) -> Feature:
        if self.find_duplicates:
            train_duplicates, valid_duplicates, intersection_duplicates = self._find_duplicates()
            self.stats["train"].duplicate_images_in_train = train_duplicates
            self.stats["val"].duplicate_images_in_validation = valid_duplicates
            self.stats["train"].duplicate_images_in_validation = intersection_duplicates
            self.stats["val"].duplicate_images_in_train = intersection_duplicates

        for basic_stats in self.stats.values():
            if basic_stats.image_count > 0:
                basic_stats.classes_in_use = len(set(basic_stats.classes))

                basic_stats.classes = np.array(basic_stats.classes)
                basic_stats.annotations_per_image = np.array(basic_stats.annotations_per_image)
                basic_stats.annotations_sizes = np.array(basic_stats.annotations_sizes)

                basic_stats.annotation_count = int(np.sum(basic_stats.annotations_per_image))
                basic_stats.images_without_annotation = np.count_nonzero(basic_stats.annotations_per_image == 0)

                basic_stats.images_resolutions = np.array(basic_stats.images_resolutions)
                basic_stats.smallest_annotations = int(np.min(basic_stats.annotations_sizes))
                basic_stats.largest_annotations = int(np.max(basic_stats.annotations_sizes))
                basic_stats.most_annotations = int(np.max(basic_stats.annotations_per_image))
                basic_stats.least_annotations = int(np.min(basic_stats.annotations_per_image))

                areas = basic_stats.images_resolutions[:, 0] * basic_stats.images_resolutions[:, 1]
                areas = areas[:, None]
                index_of_med = np.argsort(areas)[len(areas) // 2]
                basic_stats.med_image_resolution = self.format_resolution(basic_stats.images_resolutions[index_of_med][0])

                basic_stats.annotations_per_image = f"{basic_stats.annotation_count / basic_stats.image_count:.2f}"
                basic_stats.image_count = f"{basic_stats.image_count:,}"
                basic_stats.annotation_count = f"{basic_stats.annotation_count:,}"

                # To support JSON - delete arrays
                basic_stats.classes = None
                basic_stats.images_resolutions = None
                basic_stats.annotations_sizes = None

        json_res = {k: dataclasses.asdict(v) for k, v in self.stats.items()}

        feature = Feature(
            data=None,
            plot_options=None,
            json=json_res,
        )
        return feature

    def _find_duplicates(self):
        dhasher = DHash()
        train_encodings = dhasher.encode_images(self._train_image_dir)
        valid_encodings = dhasher.encode_images(self._valid_image_dir)
        train_encodings = {str(os.path.join(self._train_image_dir, k)): v for k, v in train_encodings.items()}
        valid_encodings = {str(os.path.join(self._valid_image_dir, k)): v for k, v in valid_encodings.items()}
        all_encodings = {**train_encodings, **valid_encodings}
        dups = dhasher.find_duplicates(encoding_map=all_encodings, max_distance_threshold=0)
        dups = {k: v for k, v in dups.items() if len(v) > 0}
        train_dups = []
        valid_dups = []
        intersection_dups = []
        dup_clique_heads = list(dups.keys())
        for i in range(len(dup_clique_heads)):
            dup_key = dup_clique_heads[i]
            if self._is_in_dup_clique(dup_key, train_dups) or self._is_in_dup_clique(dup_key, valid_dups) or self._is_in_dup_clique(dup_key, intersection_dups):
                continue
            dup_clique = self._make_dup_clique(dup_key, dups)
            if self._is_train_dup(dup_clique):
                train_dups.append([d for d in dup_clique if d.startswith(self._train_image_dir)])
            if self._is_valid_dup(dup_clique):
                valid_dups.append([d for d in dup_clique if d.startswith(self._valid_image_dir)])
            if self._is_intersection_dup(dup_clique):
                intersection_dups.append(dup_clique)

        return train_dups, valid_dups, intersection_dups
        # for dup in dups[dup_key]:
        #     if dup in dups.keys():
        #         del dups[dup]

    @property
    def title(self) -> str:
        return "General Statistics"

    @property
    def description(self) -> str:
        return self.template.render(**self.stats)

    @staticmethod
    def format_resolution(array: np.ndarray) -> str:
        return "x".join([str(int(x)) for x in array])
