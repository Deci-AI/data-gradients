import collections
import dataclasses
from typing import List

import numpy as np
from jinja2 import Template

from data_gradients.assets import assets
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors import AbstractFeatureExtractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from data_gradients.utils.data_classes.data_samples import ClassificationSample


@dataclasses.dataclass
class ClassificationBasicStatistics:

    num_samples: int = 0
    classes_count: int = 0
    classes_in_use: int = 0
    classes: List[int] = dataclasses.field(default_factory=list)
    images_resolutions: List[int] = dataclasses.field(default_factory=list)
    med_image_resolution: int = 0


@register_feature_extractor()
class ClassificationSummaryStats(AbstractFeatureExtractor):
    """Extracts general summary statistics from images."""

    def __init__(self):
        super().__init__()
        self.stats = {"train": ClassificationBasicStatistics(), "val": ClassificationBasicStatistics()}

        self.template = Template(source=assets.html.basic_info_fe_classification)

    def update(self, sample: ClassificationSample):

        basic_stats = self.stats[sample.split]

        height, width = sample.image.shape[:2]
        basic_stats.images_resolutions.append([height, width])
        basic_stats.num_samples += 1
        basic_stats.classes_count = len(sample.class_names)
        basic_stats.classes.append(sample.class_id)

    def aggregate(self) -> Feature:
        for basic_stats in self.stats.values():
            if basic_stats.num_samples > 0:
                basic_stats.classes_in_use = len(set(basic_stats.classes))

                images_resolutions = np.array(basic_stats.images_resolutions)
                areas = images_resolutions[:, 0] * images_resolutions[:, 1]
                index_of_med = np.argsort(areas)[len(areas) // 2]
                basic_stats.med_image_resolution = self.format_resolution(images_resolutions[index_of_med])
                basic_stats.num_samples = int(basic_stats.num_samples)

                # To support JSON - delete arrays
                basic_stats.classes = None

        json_res = {k: dataclasses.asdict(v) for k, v in self.stats.items()}

        feature = Feature(
            data=None,
            plot_options=None,
            json=json_res,
        )
        return feature

    @property
    def title(self) -> str:
        return "General Statistics"

    @property
    def description(self) -> str:
        return self.template.render(**self.stats)

    @staticmethod
    def format_resolution(array: np.ndarray) -> str:
        return "x".join([str(int(x)) for x in array])
