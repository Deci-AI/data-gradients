import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Union, Tuple, Mapping, List
from abc import ABC
import torch

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


@dataclass
class DataConfig(ABC):
    images_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None
    labels_extractor: Optional[Callable[[SupportedData], torch.Tensor]] = None

    answers_cache: Dict[str, Any] = field(default_factory=dict, init=False)

    def _answer_question_with_cache(self, caching_key: str, question: Optional[Question], hint: str = "") -> Any:
        """Method responsible for the whole logic of the class. Read class description for more information.

        :param caching_key: Name of the parameter (as passed in the `__init__` and saved in the cache file)s
        :param question:    Question to ask the user for the parameter. This is only used when the parameter was not set in the `__init__` and was
                                not found in the cache.
        :param hint:        Hint to display to the user. This is only displayed when asking a question to the user, and aims at providing extra context,
                                such as showing a sample of data, to help the user answer the question.
        """
        if caching_key not in self.answers_cache:
            self.answers_cache[caching_key] = ask_user(question.question, options=list(question.options.keys()), optional_description=hint)
        cached_answer = self.answers_cache[caching_key]
        return question.options.get(cached_answer)

    def get_images_extractor(self, question: Question, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        if self.images_extractor is None:
            self.images_extractor = self._answer_question_with_cache(caching_key="images_extractor", question=question, hint=hint)
        return self.images_extractor

    def get_labels_extractor(self, question: Question, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        if self.labels_extractor is None:
            self.labels_extractor = self._answer_question_with_cache(caching_key="labels_extractor", question=question, hint=hint)
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
            self.is_label_first = self._answer_question_with_cache(caching_key="is_label_first", question=question, hint=hint)
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
            self.xyxy_converter = self._answer_question_with_cache(caching_key="xyxy_converter", question=question, hint=hint)
        return self.xyxy_converter
