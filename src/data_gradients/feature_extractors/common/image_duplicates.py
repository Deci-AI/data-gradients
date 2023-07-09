from collections import Iterable
from typing import List
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from imagededup.methods import DHash

import os

from data_gradients.utils.data_classes import ImageSample


@register_feature_extractor()
class ImageDuplicates(AbstractFeatureExtractor):
    """Extracts the distribution image Height and Width."""

    def __init__(self):
        super().__init__()
        self.train_dups = None
        self.valid_dups = None
        self.intersection_dups = None
        self.train_image_dir = None
        self.valid_image_dir = None

    def prep_for_duplicated_detection(self, train_data: Iterable, valid_data: Iterable):
        # TODO: ADD AUTOMATIC EXTRACTION FOR SG DATASETS
        self.train_image_dir = self._get_image_dir("train")
        if valid_data is not None:
            self.valid_image_dir = self._get_image_dir("validation")

    def _get_image_dir(self, split: str) -> str:
        p = input(f"Image duplicates extraction: please enter the full path to the directory containing all {split} images >>> \n")
        if not os.path.exists(p):
            raise ValueError(f"Path to the directory containing all {split} images does not exist.")
        return p

    def update(self, sample: ImageSample):
        pass

    def _find_duplicates(self):
        dhasher = DHash()
        train_encodings = dhasher.encode_images(self.train_image_dir)
        valid_encodings = dhasher.encode_images(self.valid_image_dir)

        train_encodings = {str(os.path.join(self.train_image_dir, k)): v for k, v in train_encodings.items()}
        valid_encodings = {str(os.path.join(self.valid_image_dir, k)): v for k, v in valid_encodings.items()}
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
                train_dups.append([d for d in dup_clique if d.startswith(self.train_image_dir)])
            if self._is_valid_dup(dup_clique):
                valid_dups.append([d for d in dup_clique if d.startswith(self.valid_image_dir)])
            if self._is_intersection_dup(dup_clique):
                intersection_dups.append(dup_clique)

        self.train_dups, self.valid_dups, self.intersection_dups = train_dups, valid_dups, intersection_dups
        self.train_dups_appearences = self._count_dir_dup_appearences(self.train_dups, self.train_image_dir)
        self.validation_dups_appearences = self._count_dir_dup_appearences(self.valid_dups, self.valid_image_dir)
        self.intersection_train_appearnces = self._count_dir_dup_appearences(self.intersection_dups, self.train_image_dir)
        self.intersection_val_appearnces = self._count_dir_dup_appearences(self.intersection_dups, self.valid_image_dir)

    def _is_in_dup_clique(self, sample, dup_clique_list):
        return any([sample in d for d in dup_clique_list])

    def _make_dup_clique(self, dup_key, dups):
        dup_clique = [dup_key] + dups[dup_key]
        return dup_clique

    def _is_train_dup(self, dup_clique):
        return len([d for d in dup_clique if d.startswith(self.train_image_dir)]) > 1

    def _is_valid_dup(self, dup_clique):
        return len([d for d in dup_clique if d.startswith(self.valid_image_dir)]) > 1

    def _is_intersection_dup(self, dup_clique):
        return len([d for d in dup_clique if d.startswith(self.train_image_dir)]) > 0 and len([d for d in dup_clique if d.startswith(self.valid_image_dir)]) > 0

    def _count_dup_appearences(self, dups):
        print("App: " + str(dups))
        return sum([len(d) for d in dups])

    def _count_dir_dup_appearences(self, dups, dir):
        return self._count_dup_appearences(list(map(lambda dup: [d for d in dup if d.startswith(dir)], dups)))

    def aggregate(self) -> Feature:
        self._find_duplicates()
        feature = Feature(
            data=None,
            plot_options=None,
            json={"Train duplicates": self.train_dups, "Validation duplicates": self.valid_dups, "Intersection duplicates": self.intersection_dups},
        )
        return feature

    @property
    def title(self) -> str:
        return "Image Duplicates"

    @property
    def description(self) -> str:
        desc = self._get_split_description(self.train_dups, "Train", self.train_dups_appearences)
        if self.valid_image_dir is not None:
            desc += self._get_split_description(self.valid_dups, "Validation", self.validation_dups_appearences)
            desc += (
                f"<br />There are {len(self.intersection_dups)} duplicates between train and validation,"
                f" appearing {self.intersection_train_appearnces} times in the train image directory,"
                f" and {self.intersection_val_appearnces} times in the validation image directory."
            )
        return desc

    def _get_split_description(self, dups: List, split: str, appearences: int):
        desc = f"{split} duplicated images:<br /> <br /> There are {len(dups)} duplicated images.<br />"
        if len(dups) > 0:
            desc = desc.replace(".<br />", f" appearing {appearences} times across the dataset.<br /><br />")
        return desc
