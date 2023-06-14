import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Union, Tuple, Mapping, List
from abc import ABC
import torch
import json

from data_gradients.batch_processors.adapters.tensor_extractor import NestedDataLookup
from data_gradients.utils.detection import XYXYConverter
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


def ask_question(question: Optional[Question], hint: str = "") -> Any:
    """Method responsible for the whole logic of the class. Read class description for more information.

    :param question:    Question to ask the user for the parameter. This is only used when the parameter was not set in the `__init__` and was
                            not found in the cache.
    :param hint:        Hint to display to the user. This is only displayed when asking a question to the user, and aims at providing extra context,
                            such as showing a sample of data, to help the user answer the question.
    """
    if question is not None:
        answer = ask_user(question.question, options=list(question.options.keys()), optional_description=hint)
        return question.options[answer]


@dataclass
class DataConfig(ABC):
    images_extractor: Optional[Union[str, Callable[[SupportedData], torch.Tensor]]] = None
    labels_extractor: Optional[Union[str, Callable[[SupportedData], torch.Tensor]]] = None

    def get_images_extractor(self, question: Optional[Question] = None, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        if self.images_extractor is None:
            self.images_extractor = ask_question(question=question, hint=hint)

        if isinstance(self.images_extractor, str):
            return NestedDataLookup(self.images_extractor)
        else:
            return self.images_extractor

    def get_labels_extractor(self, question: Optional[Question] = None, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        if self.labels_extractor is None:
            self.labels_extractor = ask_question(question=question, hint=hint)

        if isinstance(self.labels_extractor, str):
            return NestedDataLookup(self.labels_extractor)
        else:
            return self.labels_extractor

    def save_to_json(self, path: str):

        if not path.endswith(".json"):
            raise ValueError(f"`{path}` should end with `.json`")

        with open(path, "r") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load_from_json(cls, path: str):
        if not path.endswith(".json"):
            raise ValueError(f"`{path}` should end with `.json`")

        # with open(path, "r") as f:
        #     data = json.load(f)
        #
        return cls

    @classmethod
    def from_json(cls, json_dict: Dict) -> "DataConfig":
        return cls(**json_dict)

    def to_json(self) -> Dict[str, Union[str, bool, None]]:
        json_dict = {
            "images_extractor": self.images_extractor if isinstance(self.images_extractor, str) else None,
            "labels_extractor": self.labels_extractor if isinstance(self.labels_extractor, str) else None,
        }
        return json_dict

    def overwrite_missing_params(self, data: Dict):
        # TODO: add Checks
        if self.images_extractor is None:
            self.images_extractor = data.get("images_extractor")

        if self.labels_extractor is None:
            self.labels_extractor = data.get("labels_extractor")


@dataclass
class SegmentationDataConfig(DataConfig):
    pass


@dataclass
class DetectionDataConfig(DataConfig):
    is_label_first: Optional[bool] = None
    xyxy_converter: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None

    def get_is_label_first(self, hint: str = "") -> bool:
        if self.is_label_first is None:
            question = Question(
                question="Which comes first in your annotations, the class id or the bounding box?",
                options={
                    "Label comes first (e.g. [class_id, x1, y1, x2, y2])": True,
                    "Bounding box comes first (e.g. [x1, y1, x2, y2, class_id])": False,
                },
            )
            self.is_label_first = ask_question(question=question, hint=hint)
        return self.is_label_first

    def get_xyxy_converter(self, hint: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
        if self.xyxy_converter is None:
            question = Question(
                question="What is the format of the bounding boxes?",
                options=XYXYConverter.get_available_options(),
            )
            self.xyxy_converter = ask_question(question=question, hint=hint)

        if isinstance(self.xyxy_converter, str):
            return XYXYConverter(self.xyxy_converter)
        else:
            return self.xyxy_converter

    def to_json(self) -> Dict[str, Union[str, bool, None]]:
        json_dict = {
            **super().to_json(),
            "is_label_first": self.is_label_first,
            "xyxy_converter": self.xyxy_converter if isinstance(self.xyxy_converter, str) else None,
        }
        return json_dict

    def overwrite_missing_params(self, data: Dict):
        super().overwrite_missing_params(data)
        if self.is_label_first is None:
            self.is_label_first = data.get("is_label_first")
        if self.xyxy_converter is None:
            self.xyxy_converter = data.get("xyxy_converter")
