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

JSONValue = Union[str, int, float, bool, None, Dict[str, Union["JSONValue", List["JSONValue"]]]]
JSONDict = Dict[str, JSONValue]


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
class CachableParam:
    value: Optional[Any]
    name: Optional[str]


class CachableTensorExtractor:
    @staticmethod
    def resolve(tensor_extractor: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> CachableParam:
        if tensor_extractor is None:
            return CachableParam(value=None, name=None)
        elif isinstance(tensor_extractor, str):
            return CachableParam(NestedDataLookup(tensor_extractor), tensor_extractor)
        elif isinstance(tensor_extractor, Callable):
            return CachableParam(value=tensor_extractor, name=str(tensor_extractor))
        else:
            raise TypeError(f"Extractor type `{type(tensor_extractor)}` not supported!")


class CachableXYXYConverter:
    @staticmethod
    def resolve(xyxy_converter: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> CachableParam:
        if xyxy_converter is None:
            return CachableParam(value=None, name=None)
        elif isinstance(xyxy_converter, str):
            return CachableParam(XYXYConverter(xyxy_converter), xyxy_converter)
        elif isinstance(xyxy_converter, Callable):
            return CachableParam(value=xyxy_converter, name=str(xyxy_converter))
        else:
            raise TypeError(f"`xyxy_converter` type `{type(xyxy_converter)}` not supported!")


@dataclass
class DataConfig(ABC):
    images_extractor: Union[None, str, Callable[[SupportedData], torch.Tensor]] = None
    labels_extractor: Union[None, str, Callable[[SupportedData], torch.Tensor]] = None

    @classmethod
    def from_json(cls, json_dict: JSONDict) -> "DataConfig":
        try:
            return cls(**json_dict)
        except TypeError as e:
            raise TypeError(f"{e}\n\t => Could not instantiate `{cls.__name__}` from json.") from e

    def to_json(self) -> JSONDict:
        json_dict = {
            "images_extractor": CachableTensorExtractor.resolve(self.images_extractor).name,
            "labels_extractor": CachableTensorExtractor.resolve(self.labels_extractor).name,
        }
        return json_dict

    @classmethod
    def load_from_json(cls, path: str) -> "DataConfig":

        if not path.endswith(".json"):
            raise ValueError(f"`{path}` should end with `.json`")

        with open(path, "r") as f:
            json_dict = json.load(f)

        return cls.from_json(json_dict)

    def save_to_json(self, path: str) -> None:

        if not path.endswith(".json"):
            raise ValueError(f"`{path}` should end with `.json`")

        with open(path, "r") as f:
            json.dump(self.to_json(), f)

    def get_images_extractor(self, question: Optional[Question] = None, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        if self.images_extractor is None:
            self.images_extractor = ask_question(question=question, hint=hint)
        return CachableTensorExtractor.resolve(tensor_extractor=self.images_extractor).value

    def get_labels_extractor(self, question: Optional[Question] = None, hint: str = "") -> Callable[[SupportedData], torch.Tensor]:
        if self.labels_extractor is None:
            self.labels_extractor = ask_question(question=question, hint=hint)
        return CachableTensorExtractor.resolve(tensor_extractor=self.labels_extractor).value

    def overwrite_missing_params(self, data: JSONDict):
        if self.images_extractor is None:
            self.images_extractor = CachableTensorExtractor.resolve(data.get("images_extractor")).value
        if self.labels_extractor is None:
            self.labels_extractor = CachableTensorExtractor.resolve(data.get("labels_extractor")).value


@dataclass
class SegmentationDataConfig(DataConfig):
    pass


@dataclass
class DetectionDataConfig(DataConfig):
    is_label_first: Union[None, bool] = None
    xyxy_converter: Union[None, str, Callable[[torch.Tensor], torch.Tensor]] = None

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
        return CachableTensorExtractor.resolve(self.xyxy_converter).value

    def to_json(self) -> JSONDict:
        json_dict = {
            **super().to_json(),
            "is_label_first": self.is_label_first,
            "xyxy_converter": CachableTensorExtractor.resolve(self.xyxy_converter).name,
        }
        return json_dict

    def overwrite_missing_params(self, data: JSONDict) -> None:
        super().overwrite_missing_params(data)
        if self.is_label_first is None:
            self.is_label_first = data.get("is_label_first")
        if self.xyxy_converter is None:
            self.xyxy_converter = CachableXYXYConverter.resolve(data.get("xyxy_converter")).value
