import os
import logging

import platformdirs
import torch
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union, List

import data_gradients
from data_gradients.dataset_adapters.config.questions import FixedOptionsQuestion, OpenEndedQuestion, text_to_yellow
from data_gradients.dataset_adapters.config.caching_utils import TensorExtractorResolver, XYXYConverterResolver
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType, JSONDict
from data_gradients.utils.detection import XYXYConverter
from data_gradients.utils.utils import safe_json_load, write_json
from data_gradients.utils.data_classes.data_samples import str
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_default_cache_dir() -> str:
    return platformdirs.user_cache_dir("DataGradients", "Deci")


@dataclass
class DataConfig(ABC):
    """Data class for handling Dataset/Dataloader configuration.

    Works as a regular dataclass, but with some additional features:
        - Getter functions that ask the user for information if this information was not provided yet.
        - Caching system, that supports saving and loading of any non-callable attribute.
            Also supports saving and loading from callable defined within DataGradients.
    """

    images_extractor: Union[None, str, Callable[[SupportedDataType], torch.Tensor]] = None
    labels_extractor: Union[None, str, Callable[[SupportedDataType], torch.Tensor]] = None
    is_batch: Union[None, bool] = None

    image_format: Union[None, str] = None

    n_classes: Union[None, int] = None
    class_names: Union[None, List[str]] = None
    class_names_to_use: Union[None, List[str]] = None

    cache_path: Optional[str] = None

    def __post_init__(self):
        # Once the object is initialized, we check if the cache is activated or not.
        if self.cache_path is not None and os.path.isfile(self.cache_path):
            self.update_from_cache_file()
        else:
            logger.info(f"Cache deactivated for `{self.__class__.__name__}`.")

    def update_from_cache_file(self):
        """Update the values that are not set yet, using the cache file."""
        if self.cache_path is not None and os.path.isfile(self.cache_path):
            self._fill_missing_params_with_cache(self.cache_path)

    def dump_cache_file(self):
        """Save the current state to the cache file."""
        if self.cache_path is not None:
            self.write_to_json(self.cache_path)

    def get_caching_info(self) -> str:
        """Get information about the status of the caching."""
        if self.cache_path is None:
            return f"`{self.__class__.__name__}` cache is not enabled because `cache_path={self.cache_path}` was not set."
        return f"`{self.__class__.__name__}` cache is set to `cache_path={self.cache_path}`."

    @classmethod
    def load_from_json(cls, cache_path: str) -> "DataConfig":
        """Load an instance of DataConfig directly from a cache file.
        :param cache_path: Path to the cache file. This should include ".json" extension.
        :return: An instance of DataConfig loaded from the cache file.
        """
        try:
            return cls(**cls._load_json_dict(path=cache_path))
        except TypeError as e:
            raise TypeError(f"{e}\n\t => Could not load `{cls.__name__}` from cache.") from e

    @staticmethod
    def _load_json_dict(path: str) -> Dict:
        """Load cache if available."""
        json_dict = safe_json_load(path=path)
        metadata = json_dict.get("metadata", {})
        if not json_dict:
            return {}
        elif metadata.get("__version__") == data_gradients.__version__:
            return json_dict.get("attributes", {})
        else:
            logger.info(
                f"{path} was not loaded from cache due to data-gradients missmatch between cache and current version"
                f"cache={json_dict.get('__version__')}!={data_gradients.__version__}=installed"
            )
            return {}

    def write_to_json(self, cache_path: str):
        """Save the serializable representation of the class to a .json file.
        :param cache_path: Full path to the cache file. This should end with ".json" extension
        """
        if not cache_path.endswith(".json"):
            raise ValueError(f"`{cache_path}` should end with `.json`")

        json_dict = {"metadata": {"__version__": data_gradients.__version__}, "attributes": self.to_json()}
        write_json(json_dict=json_dict, path=cache_path)

    def to_json(self) -> JSONDict:
        """Convert the dataclass into a serializable representation that can be saved and loaded safely.
        :return: JSON like dictionary, that can be used to create a new instance of the object.
        """
        json_dict = {
            "images_extractor": TensorExtractorResolver.to_string(self.images_extractor),
            "labels_extractor": TensorExtractorResolver.to_string(self.labels_extractor),
            "is_batch": self.is_batch,
            "image_format": self.image_format.value,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "class_names_to_use": self.class_names_to_use,
        }
        return json_dict

    @property
    def is_completely_initialized(self) -> bool:
        """Check if all the attributes are set or not."""
        return all(v is not None for v in self.to_json().values())

    def _fill_missing_params_with_cache(self, path: str):
        """Load an instance of DataConfig directly from a cache file.
        :param path: Full path of the cache file. This should end with ".json" extension.
        :return: An instance of DataConfig loaded from the cache file.
        """
        cache_dict = self._load_json_dict(path=path)
        if cache_dict:
            self._fill_missing_params(json_dict=cache_dict)

    def _fill_missing_params(self, json_dict: JSONDict):
        """Overwrite every attribute that is equal to `None`.
        This is the safe way of loading cache, since it will prioritize attributes already set by the user.

        :param json_dict: JSON like dictionary. It's values will overwrite the attributes if these attributes are None
        """
        if self.images_extractor is None:
            self.images_extractor = json_dict.get("images_extractor")
        if self.labels_extractor is None:
            self.labels_extractor = json_dict.get("labels_extractor")
        if self.is_batch is None:
            self.is_batch = json_dict.get("is_batch")
        if self.n_classes is None:
            self.n_classes = json_dict.get("n_classes")
        if self.class_names is None:
            self.class_names = json_dict.get("class_names")
        if self.class_names_to_use is None:
            self.class_names_to_use = json_dict.get("class_names_to_use")
        if self.image_format is None:
            self.image_format = str(json_dict.get("image_format", str.UNKNOWN.value))  # Load the string and convert to Enum

    def get_images_extractor(self, question: Optional[FixedOptionsQuestion] = None, hint: str = "") -> Callable[[SupportedDataType], torch.Tensor]:
        if self.images_extractor is None:
            self.images_extractor = question.ask(hint=hint)
        return TensorExtractorResolver.to_callable(tensor_extractor=self.images_extractor)

    def get_labels_extractor(self, question: Optional[FixedOptionsQuestion] = None, hint: str = "") -> Callable[[SupportedDataType], torch.Tensor]:
        if self.labels_extractor is None:
            self.labels_extractor = question.ask(hint=hint)
        return TensorExtractorResolver.to_callable(tensor_extractor=self.labels_extractor)

    def get_is_batch(self, hint: str = "") -> bool:
        if self.is_batch is None:
            question = FixedOptionsQuestion(
                question="Does your dataset provide a batch or a single sample?",
                options={
                    "Batch of Samples (e.g. torch Dataloader)": True,
                    "Single Sample (e.g. torch Dataset)": False,
                },
            )
            self.is_batch: bool = question.ask(hint=hint)
        return self.is_batch

    def get_class_names(self) -> List[str]:
        if self.class_names is None:
            self._setup_class_related_params()
        return self.class_names

    def get_n_classes(self) -> int:
        if self.n_classes is None:
            self._setup_class_related_params()
        return self.n_classes

    def get_class_names_to_use(self) -> List[str]:
        if self.class_names_to_use is None:
            self._setup_class_related_params()
        return self.class_names_to_use

    def _setup_class_related_params(self):
        """Resolve class related params.

        All the parameters are set up together because strongly related - knowing only `class_names` or `n_classes` is enough to set the values of the other 2.
        """
        self.class_names = resolve_class_names(class_names=self.class_names, n_classes=self.n_classes)
        self.n_classes = len(self.class_names)
        self.class_names_to_use = resolve_class_names_to_use(class_names=self.class_names, class_names_to_use=self.class_names_to_use)

    def get_image_format(self, hint: str = "") -> str:
        if self.image_format is None:
            pass
            # from functools import partial

            # TODO: ask from OUTSIDE
            # question = OpenEndedQuestion(
            #     question="What is the format of your images? (R, G, B, L, O)",
            #     validation=partial(validate_channels, n_channels=3),  # TODO
            # )
            # self.image_format = question.ask(hint=hint)
        return self.image_format


class ImageFormat(Enum):
    RGB = "RGB"
    BGR = "BGR"
    GRAYSCALE = "GRAYSCALE"
    LAB = "LAB"


class ImageChannels:
    def __init__(self, channels_str: str, n_channels: int):
        self.channels_str = channels_str
        self.n_channels = n_channels
        self.validate_channels(channels_str=channels_str, n_channels=n_channels)

    @staticmethod
    def validate_channels(channels_str: str, n_channels: int) -> bool:
        """
        Validate the channel string based on the following rules:
        1. The string includes at most one 'R', one 'G', and one 'B'.
        2. The string includes at most one 'L'.
        3. 'R', 'G', and 'B' are not used along with 'L'.
        4. No characters outside the set ['R', 'G', 'B', 'L', 'O'] are allowed.
        5. The string is not empty.
        6. The string length matches n_channels.

        :param channels_str: The channel string to validate.
        :param n_channels: Expected number of channels.
        :return: True if the string is valid, raises ValueError otherwise.
        """

        # Check for rule 5
        if not channels_str:
            raise ValueError("The channel string is empty.")

        # Check for rule 6
        if len(channels_str) != n_channels:
            raise ValueError(f"The channel string length ({len(channels_str)}) does not match the expected number of channels ({n_channels}).")

        # Check for rule 1 and 2
        for ch in ["R", "G", "B", "L"]:
            if channels_str.count(ch) > 1:
                raise ValueError(f"Multiple occurrences of '{ch}' found. Each channel should appear at most once.")

        # Check for rule 3
        if "L" in channels_str and any(ch in channels_str for ch in ["R", "G", "B"]):
            raise ValueError("Grayscale ('L') cannot be used in combination with RGB channels.")

        # Check for rule 4
        if any(ch not in ["R", "G", "B", "L", "O"] for ch in channels_str):
            invalid_chars = ", ".join(set(ch for ch in channels_str if ch not in ["R", "G", "B", "L", "O"]))
            raise ValueError(f"Invalid characters in channel string: {invalid_chars}. Allowed characters are: R, G, B, L, O.")

        return True

    @property
    def cv2_format(self) -> ImageFormat:
        pass

    @property
    def _is_lab(self) -> bool:
        return "LAB" in self.channels_str

    @property
    def _is_grayscale(self) -> bool:
        return "L" in self.channels_str and not self._is_lab

    def get_visualization_channel_idx(self):
        """
        Retrieve the order and indices of color channels, either RGB for RGB/BGR or
        """
        if self._include_r_g_b_channels:
            return

        return [(ch, idx) for idx, ch in enumerate(self.channels_str) if ch in "RGB"]

    def convert_image_to_rgb(self, image):
        import cv2  # Assuming you're using OpenCV

        format_type = self.cv2_format

        if format_type == ImageFormat.RGB:
            return image[:, :, [self.channels_str.index(ch) for ch in "RGB"]]

        elif format_type == ImageFormat.BGR:
            bgr_image = image[:, :, [self.channels_str.index(ch) for ch in "BGR"]]
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        elif format_type == ImageFormat.GRAYSCALE:
            grayscale_image = image[:, :, self.channels_str.index("L")]
            return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

        elif format_type == ImageFormat.LAB:
            lab_image = image[:, :, [self.channels_str.index(ch) for ch in "LAB"]]
            return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

        else:
            raise ValueError("Cannot convert the given image format to RGB.")


@dataclass
class ClassificationDataConfig(DataConfig):
    pass


@dataclass
class SegmentationDataConfig(DataConfig):
    pass


@dataclass
class DetectionDataConfig(DataConfig):
    is_label_first: Union[None, bool] = None
    xyxy_converter: Union[None, str, Callable[[torch.Tensor], torch.Tensor]] = None

    def to_json(self) -> JSONDict:
        json_dict = {
            **super().to_json(),
            "is_label_first": self.is_label_first,
            "xyxy_converter": XYXYConverterResolver.to_string(self.xyxy_converter),
        }
        return json_dict

    def _fill_missing_params(self, json_dict: JSONDict):
        super()._fill_missing_params(json_dict=json_dict)
        if self.is_label_first is None:
            self.is_label_first = json_dict.get("is_label_first")
        if self.xyxy_converter is None:
            self.xyxy_converter = json_dict.get("xyxy_converter")

    def get_is_label_first(self, hint: str = "") -> bool:
        if self.is_label_first is None:
            question = FixedOptionsQuestion(
                question=f"{text_to_yellow('Which comes first')} in your annotations, the class id or the bounding box?",
                options={
                    "Label comes first (e.g. [class_id, x1, y1, x2, y2])": True,
                    "Bounding box comes first (e.g. [x1, y1, x2, y2, class_id])": False,
                },
            )
            self.is_label_first: bool = question.ask(hint=hint)
        return self.is_label_first

    def get_xyxy_converter(self, hint: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
        if self.xyxy_converter is None:
            question = FixedOptionsQuestion(
                question=f"What is the {text_to_yellow('bounding box format')}?",
                options=XYXYConverter.get_available_options(),
            )
            self.xyxy_converter = question.ask(hint=hint)
        return XYXYConverterResolver.to_callable(self.xyxy_converter)


def resolve_class_names(class_names: List[str], n_classes: int) -> List[str]:
    """Ensure that either `class_names` or `n_classes` is specified, but not both. Return the list of class names that will be used."""
    if n_classes and class_names and (len(class_names) != n_classes):
        raise RuntimeError(f"`len(class_names)={len(class_names)} != n_classes`.")
    elif n_classes is None and class_names is None:

        def _represents_int(s: str) -> bool:
            """Check if a string represents an integer."""
            try:
                int(s)
            except ValueError:
                return False
            else:
                return True

        question = OpenEndedQuestion(
            question="How many classes does your dataset include?",
            validation=lambda answer: _represents_int(answer) and int(answer) > 0,
        )
        n_classes = int(question.ask())

    return class_names or list(map(str, range(n_classes)))


def resolve_class_names_to_use(class_names: List[str], class_names_to_use: List[str]) -> List[str]:
    """Define `class_names_to_use` from `class_names` if it is specified. Otherwise, return the list of class names that will be used."""
    if class_names_to_use:
        invalid_class_names_to_use = set(class_names_to_use) - set(class_names)
        if invalid_class_names_to_use != set():
            raise RuntimeError(f"You defined `class_names_to_use` with classes that are not listed in `class_names`: {invalid_class_names_to_use}")
    return class_names_to_use or class_names
