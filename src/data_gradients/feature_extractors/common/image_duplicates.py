from typing import List, Optional, Iterable
from data_gradients.common.registry.registry import register_feature_extractor
from data_gradients.feature_extractors.abstract_feature_extractor import AbstractFeatureExtractor
from data_gradients.feature_extractors.abstract_feature_extractor import Feature
from imagededup.methods import DHash

import os

from data_gradients.utils.data_classes import ImageSample


@register_feature_extractor()
class ImageDuplicates(AbstractFeatureExtractor):
    """
    Extracts image duplicates, in the directories train_image_dir, valid_image_dir (when present) and their intersection.

    Under the hood, uses Difference Hashing (http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)
     and considers duplicates if and only if they have the exact same hash code. This means that regardless of color format
      (i.e BGR, RGB greyscale) duplicates will be found, but might result (rarely) in false positives.

    Attributes:
        train_image_dir: str, The directory containing all train images. When None, will ask the user using prompt for input.

        valid_image_dir: str. Ignored when val_data of the AbstractManager is None. The directory containing all
         valid images. When None, will ask the user using prompt for input.


        The following attributes are populated after calling self.aggreagate():

            self.train_dups: List[List[str]], a list of all image duplicate paths inside train_image_dir.

            self.valid_dups: List[List[str]], a list of all image duplicate paths inside valid_image_dir.

            self.intersection_dups: List[List[str]], a list of all image duplicate paths, that are duplicated in
             valid_image_dir and train_image_dir (i.e images that appear in train and validation).

            train_dups_appearences: int, total image count of duplicated images in train_image_dir.

            validation_dups_appearences, int, total image count of duplicated images in valid_image_dir.

            intersection_train_appearnces, int, total image count in train_image_dir that appear in valid_image_dir.

            intersection_val_appearnces int, total image count in valid_image_dir that appear in train_image_dir.


        Example:
            After running self.aggreagte() on COCO2017 detection dataset (train_image_dir="/data/coco/images/train2017/",
             valid_image_dir=/data/coco/images/val2017/):

            self.train_dups: [['/data/coco/images/train2017/000000216265.jpg', '/data/coco/images/train2017/000000411274.jpg']...]
            self.valid_dups: [] -> no duplicates in validation
            self.intersection_dups: [['/data/coco/images/train2017/000000080010.jpg', '/data/coco/images/val2017/000000140556.jpg'],
                                        ['/data/coco/images/train2017/000000535889.jpg', '/data/coco/images/val2017/000000465129.jpg']]

            self.train_dups_appearences: 72
            self.validation_dups_appearences: 0
            self.intersection_train_appearnces: 2
            self.intersection_val_appearnces: 2

            IMPORTANT: We get len(self_train_dups) = 35, but there are 72 appearences pf duplicated images in the train directory.
                This is because we have two triplet duplicates inside our train data.


    NOTES:
         - DOES NOT TAKE IN ACCOUNT ANY DATASET INTERNAL LOGIC, SIMPLY EXTRACTS THE DUPLICATES FROM THESE DIRECTORIES.
         - If an image in the image directory can't be loaded, no duplicates are searched for the image.
         - Supported image formats: 'JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF', 'GIF', 'SVG', 'PGM', 'PBM', 'WEBP'.
    """

    def __init__(self, train_image_dir: Optional[str] = None, valid_image_dir: Optional[str] = None):
        """
        :param train_image_dir: str = None, The directory containing all train images. When None, will ask the user
            using prompt for input.

        :param valid_image_dir: str = None. Ignored when val_data of the AbstractManager is None. The directory containing all
            valid images. When None, will ask the user using prompt for input.
        """
        super().__init__()
        self.train_dups = None
        self.valid_dups = None
        self.intersection_dups = None
        self.train_image_dir = train_image_dir
        self.valid_image_dir = valid_image_dir

    def setup_data_sources(self, train_data: Iterable, val_data: Iterable):
        """
        Called in AbstractManager.__init__
        In case train_image_dir: str = None, valid_image_dir: str = None in __init__, will ask the user to pass them.

        :param train_data: Iterable, the train_data used by AbstractManager (not used at the moment, but acts as a placeholder that
            will later be used to derrive the diretory automatically according to its type - i.e ImageFolder.root etc).

        :param  val_data: Iterbale, the val_Data used by AbstractManager.
        """
        # TODO: ADD AUTOMATIC EXTRACTION FOR SG DATASETS
        if self.train_image_dir is None:
            self.train_image_dir = self._get_image_dir("train")
        if val_data is not None and self.valid_image_dir is None:
            self.valid_image_dir = self._get_image_dir("validation")

    def _get_image_dir(self, split: str) -> str:
        p = input(f"Image duplicates extraction: please enter the full path to the directory containing all {split} images >>> \n")
        if not os.path.exists(p):
            raise ValueError(f"Path to the directory containing all {split} images does not exist.")
        return p

    def update(self, sample: ImageSample):
        """
        Passed as we only find the duplicates in self.aggregate (per sample updated logic does not fit here).
        """
        pass

    def _find_duplicates(self):
        """
        Finds duplicates in self.train_image_dir, self.valid_image_dir (when present) and their intersection.
        Populates self.train_dups, self.valid_dups, self.intersection_dups and the corresponding appearences count attributes
        self.train_dups_appearences, self.validation_dups_appearences, self.intersection_train_appearnces, self.intersection_val_appearnces.
        """
        dhasher = DHash()

        # encodings = { IMAGE_X_FNAME: IMAGE_X_ENCODING,...}
        train_encodings = dhasher.encode_images(self.train_image_dir)
        valid_encodings = dhasher.encode_images(self.valid_image_dir)

        # ADD PATH PREFIXES, SO WE CAN DIFFER TRAIN/VALIDATION AFTER CALLING find_duplicates
        # AND AVOID COLLISIONS IF SAME FNAMES ARE USED BETWEEN THE DATASETS
        # encodings = { IMAGE_X_PATH: IMAGE_X_ENCODING,...}
        train_encodings = {str(os.path.join(self.train_image_dir, k)): v for k, v in train_encodings.items()}
        valid_encodings = {str(os.path.join(self.valid_image_dir, k)): v for k, v in valid_encodings.items()}

        # PASS ALL ENCODINGS
        all_encodings = {**train_encodings, **valid_encodings}

        # dups = { IMAGE_X_PATH: [DUPLICATE_OF_IMAGE_X_PATH, DUPLICATE_2_OF_IMAGE_X_PATH...]...}
        dups = dhasher.find_duplicates(encoding_map=all_encodings, max_distance_threshold=0)

        # FILTER PATHS WHERE NO DUPLICATES WERE FOUND
        dups = {k: v for k, v in dups.items() if len(v) > 0}

        train_dups = []
        valid_dups = []
        intersection_dups = []
        dup_clique_heads = list(dups.keys())

        # ITERATE THROUGH THE CLIQUE 'HEADS' (I.E SOME MEMBER REPRESENTING THE DUPLICATE CLIQUE)
        for i in range(len(dup_clique_heads)):
            dup_key = dup_clique_heads[i]

            # IF THIS CLIQUE HEAD WAS ALREADY ADDED IN PREVIOUS ITERATIONS, SKIP IT TO AVOID ADDING THE SAME CLIQUE
            # TWICE (SINCE find_duplicates WILL RETURN THE DUPLICATES IN EVERY MEMBER'S ENTRY)

            if self._is_in_dup_clique(dup_key, train_dups) or self._is_in_dup_clique(dup_key, valid_dups) or self._is_in_dup_clique(dup_key, intersection_dups):
                continue

            # CREATE A 'CLIQUE` FROM THE MEMBER AND IT'S DUPLICATES:
            # IMAGE_X_PATH: [DUPLICATE_OF_IMAGE_X_PATH, DUPLICATE_2_OF_IMAGE_X_PATH...] ->
            # [IMAGE_X_PATH, DUPLICATE_OF_IMAGE_X_PATH, DUPLICATE_2_OF_IMAGE_X_PATH...]
            dup_clique = self._make_dup_clique(dup_key, dups)

            # IF THE CLIQUE HAS AT LEAST 2 PATHS IN THE TRAIN/VALIDATION DIR -
            # ADD IT TO train_dups/valid_dups AFTER FILTERING PATHS OUTSIDE THE TRAIN/VALIDATION DIR.
            if self._is_train_dup(dup_clique):
                train_dups.append([d for d in dup_clique if d.startswith(self.train_image_dir)])
            if self._is_valid_dup(dup_clique):
                valid_dups.append([d for d in dup_clique if d.startswith(self.valid_image_dir)])

            # IF THE CLIQUE HAS IT LEAST ONE PATH IN EACH - ADD IT TO intersection_dups
            if self._is_intersection_dup(dup_clique):
                intersection_dups.append(dup_clique)

        self.train_dups, self.valid_dups, self.intersection_dups = train_dups, valid_dups, intersection_dups
        self.train_dups_appearences = self._count_dir_dup_appearences(self.train_dups, self.train_image_dir)
        self.validation_dups_appearences = self._count_dir_dup_appearences(self.valid_dups, self.valid_image_dir)
        self.intersection_train_appearnces = self._count_dir_dup_appearences(self.intersection_dups, self.train_image_dir)
        self.intersection_val_appearnces = self._count_dir_dup_appearences(self.intersection_dups, self.valid_image_dir)

    @staticmethod
    def _is_in_dup_clique(sample: str, dup_clique_list: List[List[str]]) -> bool:
        """
        Whether sample is already in some duplicated clique in dup_clique_list.

        :param sample: str, an image path
        :param dup_clique_list: List[List[str]], list of duplicate image path cliques (lists).
        """
        return any([sample in d for d in dup_clique_list])

    @staticmethod
    def _make_dup_clique(dup_key: str, dups: List[str]):
        dup_clique = [dup_key] + dups[dup_key]
        return dup_clique

    def _is_train_dup(self, dup_clique: List[str]) -> bool:
        """
        Whether the dup_clique contains duplicate image paths from train_image_dir.

        :param dup_clique: List[str], list of duplicated image paths
        """
        return len([d for d in dup_clique if d.startswith(self.train_image_dir)]) > 1

    def _is_valid_dup(self, dup_clique: List[str]) -> bool:
        """
        Whether the dup_clique contains duplicate image paths from valid_image_dir.

        :param dup_clique: List[str], list of duplicated image paths
        """
        return len([d for d in dup_clique if d.startswith(self.valid_image_dir)]) > 1

    def _is_intersection_dup(self, dup_clique: List[str]) -> bool:
        """
        Whether the dup_clique contains duplicate image paths from train_image_dir and valid_image_dir.

        :param dup_clique: List[str], list of duplicated image paths
        """
        return len([d for d in dup_clique if d.startswith(self.train_image_dir)]) > 0 and len([d for d in dup_clique if d.startswith(self.valid_image_dir)]) > 0

    @staticmethod
    def _count_dup_appearences(dups: List[List[str]]) -> int:
        """
        Counts the duplicate appearences = sum of the sizes of all duplicate cliques in dups.
        """
        return sum([len(d) for d in dups])

    def _count_dir_dup_appearences(self, dups: List[List[str]], dir: str) -> int:
        """
        Counts the duplicate appearences inside dir = sum of the sizes of all
         duplicate cliques in dups after filtering all paths not in dir.
        """
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
        if self.train_dups:
            desc = self._get_split_description(self.train_dups, "Train", self.train_dups_appearences)
            if self.valid_image_dir is not None:
                desc += self._get_split_description(self.valid_dups, "Validation", self.validation_dups_appearences)
                desc += f"\n\nThere are {len(self.intersection_dups)} duplicates between train and validation."
                if len(self.intersection_dups):
                    desc = desc.replace(
                        "train and validation.",
                        f"train and validation appearing {self.intersection_train_appearnces} times in the train image directory,"
                        f" and {self.intersection_val_appearnces} times in the validation image directory.",
                    )

            else:
                desc += "\n"
            return desc
        else:
            return (
                "Shows how may duplicate images you have in your dataset:\n"
                "- How many images in your training set are duplicate.\n"
                "- How many images in your validation set are duplicate.\n"
                "- How many images are in both your validation and training set."
            )

    def _get_split_description(self, dups: List, split: str, appearences: int) -> str:
        desc = f"<strong>{split} duplicated images</strong>:\n There are {len(dups)} duplicated images.\n"
        if len(dups) > 0:
            desc = desc.replace(".\n", f" appearing {appearences} times across the dataset.\n\n")
        return desc
