import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Union, Tuple, Mapping, List
from abc import ABC
import torch

from data_gradients.utils.detection import xywh_to_xyxy, cxcywh_to_xyxy
from data_gradients.utils.utils import ask_user

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
    :param cache_path: Path to the cache file
    :param reset_cache: If set to True, the cache will be reset. Default is False
    """

    def __init__(self, cache_path: str, reset_cache: bool = False):
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        if not reset_cache:
            try:
                with open(cache_path, "r") as f:
                    self.cache = json.load(f)
            except FileNotFoundError:
                pass

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def set(self, key, value):
        self.cache[key] = value

    def save(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)


@dataclass
class BaseInteractiveConfig(ABC):
    def __init__(
        self,
        caching_path: str,
        reset_cache: bool = False,
        images_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None,
        labels_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None,
        **kwargs,
    ):
        """
        This interactive configuration class is designed to optimize the setup process by actively interacting with the user.
        Whenever a parameter's value is accessed, the class follows a sequential strategy to decide its value:

        1. First, the class checks if a value for the parameter was passed during initialization (`__init__`).
        2. If not provided at initialization, the class then looks up the parameter in the cache.
            a. If a cached value is found, it is used.
            b. If no cached value is found, the user is prompted to provide a value, which is then cached for future use.

        This strategy ensures efficient use of provided and cached data and also allows for interactive filling of information gaps.
        The value determination is made at the moment a parameter is accessed, guaranteeing up-to-date information.

        :param caching_path:        Path to the cache file. This is where cached parameter values are stored and retrieved from.
        :param reset_cache:         If set to True, the cache is cleared and any previously stored values are discarded. Default is False.
        :param images_extractor:    (Optional) How images should be extracted from the input data iterable you provide to the Analyzer.
                                    If not set, the user will be asked when this information is needed.
        :param labels_extractor: (Optional) How labels should be extracted from the input data iterable you provide to the Analyzer.
                                    If not set, the user will be asked when this information is needed.

        Examples:

        >>> config = BaseInteractiveConfig(caching_path='path/to/cache')
        In this case, no parameter values are provided during initialization, so whenever a parameter is accessed,
        the class will look up the value in the cache file at 'path/to/cache'. If no cached value is found, the user
        will be prompted to provide the value which will then be cached for future use.

        >>> config = BaseInteractiveConfig(caching_path='path/to/cache', images_extractor=my_extractor)
        Here, the images_extractor parameter is provided during initialization. So whenever `config.images_extractor`
        is accessed, the value `my_extractor` will always be used, and no cache lookup or user prompt will occur for this parameter.
        """
        self.caching_path = caching_path

        # This includes the objects that will be used throughout the code
        self._parameters = dict(images_extractor=images_extractor, labels_extractor=labels_extractor, **kwargs)

        # This includes answers to questions, which is used only when no _parameter[param_nam] value was found.
        self.cache_answers = CacheManager(cache_path=caching_path, reset_cache=reset_cache)

    def _get_parameter(self, key: str, question: Question, hint: str = "") -> Any:
        """Method responsible for the whole logic of the class. Read class description for more information.

        :param key:         Name of the parameter (as passed in the `__init__` and saved in the cache file)s
        :param question:    Question to ask the user for the parameter. This is only used when the parameter was not set in the `__init__` and was
                                not found in the cache.
        :param hint:        Hint to display to the user. This is only displayed when asking a question to the user, and aims at providing extra context,
                                such as showing a sample of data, to help the user answer the question.
        """
        if self._parameters.get(key) is None:
            if self.cache_answers.get(key) is None:
                answer = ask_user(question.question, options=list(question.options.keys()), optional_description=hint)
                self.cache_answers.set(key, answer)
            cached_answer = self.cache_answers.get(key)
            self._parameters[key] = question.options[cached_answer]
        return self._parameters[key]

    def get_images_extractor(self, question: Question, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        return self._get_parameter(key="images_extractor", question=question, hint=hint)

    def set_images_extractor(self, image_extractor: Callable[[SupportedData], torch.Tensor], path_description: str):
        """Manually save the images_extractor to method _parameters for later use, and to the cache file for traceability.
        :param image_extractor:     The images_extractor to save.
        :param path_description:    Description of the path required to find the images from the dataset/dataloader output.
        """
        self._parameters["images_extractor"] = image_extractor
        self.cache_answers.set("images_extractor", path_description)

    def get_labels_extractor(self, question: Question, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        return self._get_parameter(key="labels_extractor", question=question, hint=hint)

    def set_labels_extractor(self, labels_extractor: Callable[[SupportedData], torch.Tensor], path_description: str):
        """Manually save the labels_extractor to method _parameters for later use, and to the cache file for traceability.
        :param labels_extractor:    The images_extractor to save.
        :param path_description:    Description of the path required to find the labels from the dataset/dataloader output.
        """
        self._parameters["labels_extractor"] = labels_extractor
        self.cache_answers.set("labels_extractor", path_description)  # This won't be used directly but improves tracability of what we do.

    def save_cache(self):
        """Save all the cached answers to the cache file."""
        self.cache_answers.save()


@dataclass
class SegmentationInteractiveConfig(BaseInteractiveConfig):
    def __init__(
        self,
        caching_path: str,
        reset_cache: bool = False,
        images_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None,
        labels_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None,
    ):
        """
        This interactive configuration class is designed to optimize the setup process by actively interacting with the user.
        Whenever a parameter's value is accessed, the class follows a sequential strategy to decide its value:

        1. First, the class checks if a value for the parameter was passed during initialization (`__init__`).
        2. If not provided at initialization, the class then looks up the parameter in the cache.
            a. If a cached value is found, it is used.
            b. If no cached value is found, the user is prompted to provide a value, which is then cached for future use.

        This strategy ensures efficient use of provided and cached data and also allows for interactive filling of information gaps.
        The value determination is made at the moment a parameter is accessed, guaranteeing up-to-date information.

        :param caching_path:        Path to the cache file. This is where cached parameter values are stored and retrieved from.
        :param reset_cache:         If set to True, the cache is cleared and any previously stored values are discarded. Default is False.
        :param images_extractor:    (Optional) How images should be extracted from the input data iterable you provide to the Analyzer.
                                    If not set, the user will be asked when this information is needed.
        :param labels_extractor: (Optional) How labels should be extracted from the input data iterable you provide to the Analyzer.
                                    If not set, the user will be asked when this information is needed.

        Examples:

        >>> config = SegmentationInteractiveConfig(caching_path='path/to/cache')
        In this case, no parameter values are provided during initialization, so whenever a parameter is accessed,
        the class will look up the value in the cache file at 'path/to/cache'. If no cached value is found, the user
        will be prompted to provide the value which will then be cached for future use.

        >>> config = SegmentationInteractiveConfig(caching_path='path/to/cache', images_extractor=my_extractor)
        Here, the images_extractor parameter is provided during initialization. So whenever `config.images_extractor`
        is accessed, the value `my_extractor` will always be used, and no cache lookup or user prompt will occur for this parameter.
        """
        super().__init__(
            caching_path=caching_path,
            reset_cache=reset_cache,
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
        )


@dataclass
class DetectionInteractiveConfig(BaseInteractiveConfig):
    def __init__(
        self,
        caching_path: str,
        reset_cache: bool = False,
        images_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None,
        labels_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None,
        is_label_first: Optional[bool] = None,
        xyxy_converter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        This interactive configuration class is designed to optimize the setup process by actively interacting with the user.
        Whenever a parameter's value is accessed, the class follows a sequential strategy to decide its value:

        1. First, the class checks if a value for the parameter was passed during initialization (`__init__`).
        2. If not provided at initialization, the class then looks up the parameter in the cache.
            a. If a cached value is found, it is used.
            b. If no cached value is found, the user is prompted to provide a value, which is then cached for future use.

        This strategy ensures efficient use of provided and cached data and also allows for interactive filling of information gaps.
        The value determination is made at the moment a parameter is accessed, guaranteeing up-to-date information.

        :param caching_path:        Path to the cache file. This is where cached parameter values are stored and retrieved from.
        :param reset_cache:         If set to True, the cache is cleared and any previously stored values are discarded. Default is False.
        :param images_extractor:    (Optional) How images should be extracted from the input data iterable you provide to the Analyzer.
                                    If not set, the user will be asked when this information is needed.
        :param labels_extractor:    (Optional) How labels should be extracted from the input data iterable you provide to the Analyzer.
                                    If not set, the user will be asked when this information is needed.
        :param is_label_first:      (Optional) If True, the labels are defined starting with the class_id (class_id, x1, y1, x2, y2), (class_id, y, x, w, h), ..
        :param xyxy_converter:      (Optional) How to convert the bounding box coordinates from the dataset format into to (x1, y1, x2, y2).

        Examples:

        >>> config = DetectionInteractiveConfig(caching_path='path/to/cache')
        In this case, no parameter values are provided during initialization, so whenever a parameter is accessed,
        the class will look up the value in the cache file at 'path/to/cache'. If no cached value is found, the user
        will be prompted to provide the value which will then be cached for future use.

        >>> config = DetectionInteractiveConfig(caching_path='path/to/cache', images_extractor=my_extractor)
        Here, the images_extractor parameter is provided during initialization. So whenever `config.images_extractor`
        is accessed, the value `my_extractor` will always be used, and no cache lookup or user prompt will occur for this parameter.
        """
        super().__init__(
            caching_path=caching_path,
            reset_cache=reset_cache,
            images_extractor=images_extractor,
            labels_extractor=labels_extractor,
            is_label_first=is_label_first,
            xyxy_converter=xyxy_converter,
        )

    def is_label_first(self, hint: str = "") -> bool:
        return self._get_parameter(key="is_label_first", question=STATIC_QUESTIONS["is_label_first"], hint=hint)

    def xyxy_converter(self, hint: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
        return self._get_parameter(key="xyxy_converter", question=STATIC_QUESTIONS["xyxy_converter"], hint=hint)
