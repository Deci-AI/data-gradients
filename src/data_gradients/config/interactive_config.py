import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from abc import ABC
import torch

from data_gradients.utils.detection import xywh_to_xyxy, cxcywh_to_xyxy
from data_gradients.utils.utils import ask_user


@dataclass
class Question:
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
    def __init__(self, cache_path, reset_cache: bool = False):
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
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
        **kwargs,
    ):
        self.caching_path = caching_path

        # This includes the objects that will be used throughout the code
        self._parameters = dict(images_extractor=images_extractor, labels_extractor=labels_extractor, **kwargs)

        # This includes answers to questions, which is used only when no _parameter[param_nam] value was found.
        self.cache_answers = CacheManager(cache_path=caching_path, reset_cache=reset_cache)

    def _get_parameter(self, key: str, question: Question, hint: str = "") -> Any:
        if self._parameters.get(key) is None:
            if self.cache_answers.get(key) is None:
                answer = ask_user(question.question, options=list(question.options.keys()), optional_description=hint)
                self.cache_answers.set(key, answer)
            cached_answer = self.cache_answers.get(key)
            self._parameters[key] = question.options[cached_answer]
        return self._parameters[key]

    def get_images_extractor(self, question: Question, hint: str = "") -> Callable:
        return self._get_parameter(key="images_extractor", question=question, hint=hint)

    def set_images_extractor(self, image_extractor: Callable, path_description: str):
        self._parameters["images_extractor"] = image_extractor
        self.cache_answers.set("images_extractor", path_description)  # This won't be used directly but improves tracability of what we do.

    def get_labels_extractor(self, question: Question, hint: str = "") -> Callable:
        return self._get_parameter(key="labels_extractor", question=question, hint=hint)

    def set_labels_extractor(self, labels_extractor: Callable, path_description: str):
        self._parameters["labels_extractor"] = labels_extractor
        self.cache_answers.set("labels_extractor", path_description)  # This won't be used directly but improves tracability of what we do.

    def save(self):
        self.cache_answers.save()


@dataclass
class SegmentationInteractiveConfig(BaseInteractiveConfig):
    def __init__(
        self,
        caching_path: str,
        reset_cache: bool = False,
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
    ):
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
        images_extractor: Optional[Callable] = None,
        labels_extractor: Optional[Callable] = None,
        is_label_first: Optional[bool] = None,
        xyxy_converter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
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

    def xyxy_converter(self, hint: str = "") -> Callable:
        return self._get_parameter(key="xyxy_converter", question=STATIC_QUESTIONS["xyxy_converter"], hint=hint)
