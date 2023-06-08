import logging
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Union, Tuple, Mapping, List
from abc import ABC
import torch

import data_gradients
from data_gradients.utils.detection import xywh_to_xyxy, cxcywh_to_xyxy
from data_gradients.utils.utils import ask_user

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)


SupportedData = Union[Tuple, List, Mapping, Tuple, List]


@dataclass
class Question:
    """Model a Question with its options
    :attr question: The question string
    :attr options: The options for the question
    """

    question: str
    options: Dict[str, Any]


# Questions That don't change based on the data
STATIC_QUESTIONS = {
    "is_label_first": Question(
        question="Which comes first in your annotations, the class id or the bounding box?",
        options={
            "Label comes first (e.g. [class_id, x1, y1, x2, y2])": True,
            "Bounding box comes first (e.g. [x1, y1, x2, y2, class_id])": False,
        },
    ),
    "xyxy_converter": Question(
        question="What is the format of the bounding boxes?",
        options={
            "xyxy: x- left, y-top, x-right, y-bottom": lambda x: x,
            "xywh: x-left, y-top, width, height": xywh_to_xyxy,
            "cxcywh: x-center, y-center, width, height": cxcywh_to_xyxy,
        },
    ),
}


class CacheManager:
    """Manage the caching of data
    :param cache_path: Path to the cache file. If not specified, the cache is not saved.
    """

    def __init__(self, cache_path: Optional[str] = None):
        self.cache: Dict[str, str] = {"version": data_gradients.__version__}
        if cache_path is not None:
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)

                # Note: Cache between versions not supported.
                if cache.get("version") == data_gradients.__version__:
                    self.cache = cache
                    logging.info(f"Using cached data from {cache_path}. If you don't want to use the cache, set `load_cache=False`.")
            except FileNotFoundError:
                pass

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def set(self, key, value):
        self.cache[key] = value


@dataclass
class DataConfig(ABC):
    images_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None
    labels_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None

    answers_cache: Dict[str, Any] = field(default_factory=dict, init=False)

    def _get_interactive_value(self, key: str, question: Optional[Question], hint: str = "", cache: Optional[CacheManager] = None) -> Any:
        """Method responsible for the whole logic of the class. Read class description for more information.

        :param key:         Name of the parameter (as passed in the `__init__` and saved in the cache file)s
        :param question:    Question to ask the user for the parameter. This is only used when the parameter was not set in the `__init__` and was
                                not found in the cache.
        :param hint:        Hint to display to the user. This is only displayed when asking a question to the user, and aims at providing extra context,
                                such as showing a sample of data, to help the user answer the question.
        """

        if self.answers_cache.get(key) is None:
            answer = ask_user(question.question, options=list(question.options.keys()), optional_description=hint)
            self.answers_cache[key] = answer
        cached_answer = self.answers_cache.get(key)
        return question.options[cached_answer]

    def get_images_extractor(self, question: Question, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        if self.images_extractor is None:
            value = self._get_interactive_value(key="images_extractor", question=question, hint=hint)
            self.images_extractor = value
        return self.images_extractor

    def get_labels_extractor(self, question: Question, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        if self.labels_extractor is None:
            value = self._get_interactive_value(key="labels_extractor", question=question, hint=hint)
            self.labels_extractor = value
        return self.labels_extractor


@dataclass
class SegmentationDataConfig(DataConfig):
    pass


@dataclass
class DetectionDataConfig(DataConfig):
    is_label_first: Optional[bool] = None
    xyxy_converter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def get_is_label_first(self, hint: str = "") -> bool:
        if self.is_label_first is None:
            question = Question(
                question="Which comes first in your annotations, the class id or the bounding box?",
                options={
                    "Label comes first (e.g. [class_id, x1, y1, x2, y2])": True,
                    "Bounding box comes first (e.g. [x1, y1, x2, y2, class_id])": False,
                },
            )
            value = self._get_interactive_value(key="is_label_first", question=question, hint=hint)
            self.is_label_first = value
        return self.is_label_first

    def get_xyxy_converter(self, hint: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
        if self.xyxy_converter is None:
            question = Question(
                question="What is the format of the bounding boxes?",
                options={
                    "xyxy: x- left, y-top, x-right, y-bottom": lambda x: x,
                    "xywh: x-left, y-top, width, height": xywh_to_xyxy,
                    "cxcywh: x-center, y-center, width, height": cxcywh_to_xyxy,
                },
            )
            value = self._get_interactive_value(key="xyxy_converter", question=question, hint=hint)
            self.xyxy_converter = value
        return self.xyxy_converter


# @dataclass
# class BaseInteractiveDataConfig(ABC):
#     def __init__(self, data_config: DataConfig, caching_path: str, load_cache: bool = True):
#         """
#         This interactive configuration class is designed to optimize the setup process by actively interacting with the user.
#         Whenever a parameter's value is accessed, the class follows a sequential strategy to decide its value:
#
#         1. First, the class checks if a value for the parameter was passed during initialization (`__init__`).
#         2. If not provided at initialization, the class then looks up the parameter in the cache.
#             a. If a cached value is found, it is used.
#             b. If no cached value is found, the user is prompted to provide a value, which is then cached for future use.
#
#         This strategy ensures efficient use of provided and cached data and also allows for interactive filling of information gaps.
#         The value determination is made at the moment a parameter is accessed, guaranteeing up-to-date information.
#
#         :param load_cache:          If True, the cache will be loaded from previous session (if there was)
#
#         Examples:
#
#         >>> config = BaseInteractiveDataConfig(report_title='experiment')
#         In this case, no parameter values are provided during initialization, so whenever a parameter is accessed,
#         the class will look up the value in the cache file at 'experiment'. If no cached value is found, the user
#         will be prompted to provide the value which will then be cached for future use.
#
#         >>> config = BaseInteractiveDataConfig(report_title='experiment', images_extractor=my_extractor)
#         Here, the images_extractor parameter is provided during initialization. So whenever `config.images_extractor`
#         is accessed, the value `my_extractor` will always be used, and no cache lookup or user prompt will occur for this parameter.
#         """
#         self._interactive_parameters = asdict(data_config)
#
#         self.answers_cache = CacheManager(cache_path=caching_path, load_cache=load_cache)
#
#     def _get_interactive_parameters(self, key: str, question: Question, hint: str = "") -> Any:
#         """Method responsible for the whole logic of the class. Read class description for more information.
#
#         :param key:         Name of the parameter (as passed in the `__init__` and saved in the cache file)s
#         :param question:    Question to ask the user for the parameter. This is only used when the parameter was not set in the `__init__` and was
#                                 not found in the cache.
#         :param hint:        Hint to display to the user. This is only displayed when asking a question to the user, and aims at providing extra context,
#                                 such as showing a sample of data, to help the user answer the question.
#         """
#         if self._interactive_parameters.get(key) is None:
#             if self.answers_cache.get(key) is None:
#                 answer = ask_user(question.question, options=list(question.options.keys()), optional_description=hint)
#                 self.answers_cache.set(key, answer)
#             cached_answer = self.answers_cache.get(key)
#             self._interactive_parameters[key] = question.options[cached_answer]
#         return self._interactive_parameters[key]
#
#     def get_images_extractor(self, question: Question, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
#         return self._get_interactive_parameters(key="images_extractor", question=question, hint=hint)
#
#     def set_images_extractor(self, image_extractor: Callable[[SupportedData], torch.Tensor], path_description: str):
#         """Manually save the images_extractor to method _parameters for later use, and to the cache file for traceability.
#         :param image_extractor:     The images_extractor to save.
#         :param path_description:    Description of the path required to find the images from the dataset/dataloader output.
#         """
#         self._interactive_parameters["images_extractor"] = image_extractor
#         self.answers_cache.set("images_extractor", path_description)
#
#     def get_labels_extractor(self, question: Question, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
#         return self._get_interactive_parameters(key="labels_extractor", question=question, hint=hint)
#
#     def set_labels_extractor(self, labels_extractor: Callable[[SupportedData], torch.Tensor], path_description: str):
#         """Manually save the labels_extractor to method _parameters for later use, and to the cache file for traceability.
#         :param labels_extractor:    The images_extractor to save.
#         :param path_description:    Description of the path required to find the labels from the dataset/dataloader output.
#         """
#         self._interactive_parameters["labels_extractor"] = labels_extractor
#         self.answers_cache.set("labels_extractor", path_description)  # This won't be used directly but improves tracability of what we do.
#
#     def save_cache(self):
#         """Save all the cached answers to the cache file."""
#         self.answers_cache.save()
#
#
# @dataclass
# class SegmentationInteractiveDataConfig(BaseInteractiveDataConfig):
#     def __init__(self, data_config: SegmentationDataConfig, caching_path: str, load_cache: bool = True):
#         if not isinstance(data_config, DetectionDataConfig):
#             raise TypeError("`data_config` must be of type `SegmentationDataConfig` when working on segmentation.")
#         super().__init__(data_config=data_config, caching_path=caching_path, load_cache=load_cache)
#
#
# @dataclass
# class DetectionInteractiveDataConfig(BaseInteractiveDataConfig):
#     def __init__(self, data_config: DetectionDataConfig, caching_path: str, load_cache: bool = True):
#         if not isinstance(data_config, DetectionDataConfig):
#             raise TypeError("`data_config` must be of type `DetectionDataConfig` when working on detection.")
#         super().__init__(data_config=data_config, caching_path=caching_path, load_cache=load_cache)
#
#     def is_label_first(self, hint: str = "") -> bool:
#         return self._get_interactive_parameters(key="is_label_first", question=STATIC_QUESTIONS["is_label_first"], hint=hint)
#
#     def xyxy_converter(self, hint: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
#         return self._get_interactive_parameters(key="xyxy_converter", question=STATIC_QUESTIONS["xyxy_converter"], hint=hint)
