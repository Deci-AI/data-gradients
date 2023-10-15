import os
import logging

import numpy as np
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
from data_gradients.utils.data_classes.image_channels import ImageChannels


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

    image_channels: Union[None, ImageChannels] = None

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
            "image_channels": None if self.image_channels is None else self.image_channels.channels_str,
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
        if self.image_channels is None:
            if json_dict.get("image_channels"):
                self.image_channels = ImageChannels.from_str(json_dict.get("image_channels"))

    def get_images_extractor(self, question: Optional[FixedOptionsQuestion] = None, hint: str = "") -> Callable[[SupportedDataType], torch.Tensor]:
        if self.images_extractor is None:
            self.images_extractor = question.ask(hint=hint)
        return TensorExtractorResolver.to_callable(tensor_extractor=self.images_extractor)

    def get_labels_extractor(self, question: Optional[FixedOptionsQuestion] = None, hint: str = "") -> Callable[[SupportedDataType], torch.Tensor]:
        if self.labels_extractor is None:
            self.labels_extractor = question.ask(hint=hint)
        return TensorExtractorResolver.to_callable(tensor_extractor=self.labels_extractor)

    def get_image_channels(self, image: Union[torch.Tensor, np.ndarray]) -> ImageChannels:

        if self.image_channels is None:

            if 1 in image.shape:
                self.image_channels = ImageChannels.from_str("G")

            elif 3 in image.shape:
                question = FixedOptionsQuestion(
                    question="In which format are your images loaded ?",
                    options={
                        "RGB": ImageChannels.from_str("RGB"),
                        "BGR": ImageChannels.from_str("BGR"),
                        "LAB": ImageChannels.from_str("LAB"),
                        "Other": ImageChannels.from_str("OOO"),
                    },
                )
                self.image_channels = question.ask()

            else:

                def _validate_image_channels(channels_str: str) -> bool:
                    if len(channels_str) not in image.shape:
                        return False
                    try:
                        ImageChannels.from_str(channels_str=channels_str)
                        print(f"image_channels_str={channels_str} is valid with {image.shape}")
                        return True
                    except ValueError:
                        return False

                question = OpenEndedQuestion(question="Please describe your image channels?", validation=_validate_image_channels)
                hint = (
                    f"Image Shape: {tuple(image.shape)}\n\n"
                    "Enter the channel format representing your image:\n"
                    "\n"
                    "  > RGB  : Red, Green, Blue\n"
                    "  > BGR  : Blue, Green, Red\n"
                    "  > G    : Grayscale\n"
                    "  > LAB  : Luminance, A and B color channels\n"
                    "\n"
                    "ADDITIONAL CHANNELS?\n"
                    "If your image contains channels other than the standard ones listed above (e.g., Depth, Heat), "
                    "prefix them with 'O'. \n"
                    "For instance:\n"
                    "  > ORGBO: Can represent (Heat, Red, Green, Blue, Depth).\n"
                    "  > OBGR:  Can represent (Alpha, Blue, Green, Red).\n"
                    "  > GO:    Can represent (Gray, Depth).\n\n"
                    f"IMPORTANT: Make sure that your answer represents all the image channels."
                )

                image_channels_str = question.ask(hint=hint)
                self.image_channels = ImageChannels.from_str(channels_str=image_channels_str)

        return self.image_channels

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


@dataclass
class ImageChannel:
    channel_names: List[str]
    channels_idx_to_visualize: List[str]
    rgb_converter: Callable[[np.ndarray], np.ndarray]


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
