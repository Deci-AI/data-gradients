import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Union, Tuple, Mapping, List
from abc import ABC
import torch
import json

from data_gradients.batch_processors.adapters.tensor_extractor import NestedDataLookup
from data_gradients.config.data.questions import Question, ask_question
from data_gradients.utils.detection import XYXYConverter

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)


SupportedData = Union[Tuple, List, Mapping, Tuple, List]

JSONValue = Union[str, int, float, bool, None, Dict[str, Union["JSONValue", List["JSONValue"]]]]
JSONDict = Dict[str, JSONValue]


@dataclass
class CachableParam:
    value: Optional[Any]
    name: Optional[str]


NON_CACHABLE_PREFIX = "[Non-cachable]"


class CacheLoadingError(Exception):
    def __init__(self, key: str, value: str):
        message = (
            f"Error while trying to load `{key}` from cache... with value `{value}`.\n"
            f"It seems that this object was passed to the `DataConfig` in the previous run.\n"
            f"Please:\n"
            f"     - Either pass the same `{key}` to the `DataConfig`.\n"
            f"     - Or disable loading config from cache.\n"
        )
        super().__init__(message)


class CachableTensorExtractor:
    """Static class"""

    @staticmethod
    def resolve(tensor_extractor: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> CachableParam:
        if tensor_extractor is None:
            return CachableParam(value=None, name=None)

        elif isinstance(tensor_extractor, str):
            if tensor_extractor.startswith(NON_CACHABLE_PREFIX):
                # The value corresponds to the cache of a custom function. THis means that the function was cached by user in previous run,
                # but he did not provide a function for this run.
                # Since we cannot build back the original function, we raise an informative exception.
                raise CacheLoadingError(key="tensor_extractor", value=tensor_extractor)
            return CachableParam(NestedDataLookup(tensor_extractor), tensor_extractor)

        elif isinstance(tensor_extractor, Callable):
            return CachableParam(value=tensor_extractor, name=f"{NON_CACHABLE_PREFIX} - {tensor_extractor}")
        else:
            raise TypeError(f"Extractor type `{type(tensor_extractor)}` not supported!")


class CachableXYXYConverter:
    @staticmethod
    def resolve(xyxy_converter: Union[None, str, Callable[[torch.Tensor], torch.Tensor]]) -> CachableParam:
        """Translate the input `xyxy_converter` into both:
            - value: Value of the `xyxy_converter` that will be used in the code.
            - name:  String representation of `xyxy_converter` when it comes to caching.

        For example:
            >> CachableXYXYConverter.resolve("xywh")                    # CachableParam(value=XYXYConverter("xywh"), name="xyxy")
            >> CachableXYXYConverter.resolve(my_custom_xyxy_comverter)  # CachableParam(value=my_custom_xyxy_comverter, name="[Non-cachable] - lambda ...")

        :param xyxy_converter: Either None, a string representation (e.g. `xywh`) or a custom callable.
        :return: Dataclass including both the value (used in the code) and the name (used in the cache).
        """
        if xyxy_converter is None:
            return CachableParam(value=None, name=None)

        elif isinstance(xyxy_converter, str):
            if xyxy_converter.startswith(NON_CACHABLE_PREFIX):
                # The value corresponds to the cache of a custom function. THis means that the function was cached by user in previous run,
                # but he did not provide a function for this run.
                # Since we cannot build back the original function, we raise an informative exception.
                raise CacheLoadingError(key="xyxy_converter", value=xyxy_converter)
            return CachableParam(XYXYConverter(xyxy_converter), xyxy_converter)

        elif isinstance(xyxy_converter, Callable):
            return CachableParam(value=xyxy_converter, name=f"{NON_CACHABLE_PREFIX} - {xyxy_converter}")
        else:
            raise TypeError(f"`xyxy_converter` type `{type(xyxy_converter)}` not supported!")


@dataclass
class DataConfig(ABC):
    """Data class for handling Dataset/Dataloader configuration.

    Works as a regular dataclass, but with some additional features:
        - Getter functions that ask the user for information if this information was not provided yet.
        - Caching system, that supports saving and loading of any non-callable attribute.
            Also supports saving and loading from callable defined within DataGradients.
    """

    images_extractor: Union[None, str, Callable[[SupportedData], torch.Tensor]] = None
    labels_extractor: Union[None, str, Callable[[SupportedData], torch.Tensor]] = None

    @classmethod
    def from_json(cls, json_dict: JSONDict) -> "DataConfig":
        """Create a new instance of the dataclass from a JSON representation.
        :param json_dict: JSON like dictionary.
        """
        try:
            return cls(**json_dict)
        except TypeError as e:
            raise TypeError(f"{e}\n\t => Could not instantiate `{cls.__name__}` from json.") from e

    def to_json(self) -> JSONDict:
        """Convert the dataclass into a serializable representation that can be saved and loaded safely.
        :return: JSON like dictionary, that can be used to create a new instance of the object.
        """
        json_dict = {
            "images_extractor": CachableTensorExtractor.resolve(self.images_extractor).name,
            "labels_extractor": CachableTensorExtractor.resolve(self.labels_extractor).name,
        }
        return json_dict

    @classmethod
    def load_from_json(cls, path: str) -> "DataConfig":
        """Load the representation of the class from a .json file.
        :param path: Path where the file is, should include ".json" extension."""

        if not path.endswith(".json"):
            raise ValueError(f"`{path}` should end with `.json`")

        with open(path, "r") as f:
            json_dict = json.load(f)

        return cls.from_json(json_dict)

    def save_to_json(self, path: str):
        """Save the serializable representation of the class to a .json file.
        :param path: Output path of the file, should include ".json" extension."""

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

    def overwrite_missing_params(self, json_dict: JSONDict):
        """Overwrite every attribute that is equal to `None`.
        This is the safe way of loading cache, since it will prioritize attributes already set by the user.

        :param json_dict: JSON like dictionary. It's values will overwrite the attributes if these attributes are None
        """
        if self.images_extractor is None:
            self.images_extractor = json_dict.get("images_extractor")
        if self.labels_extractor is None:
            self.labels_extractor = json_dict.get("labels_extractor")


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
            self.is_label_first: bool = ask_question(question=question, hint=hint)
        return self.is_label_first

    def get_xyxy_converter(self, hint: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
        if self.xyxy_converter is None:
            question = Question(
                question="What is the format of the bounding boxes?",
                options=XYXYConverter.get_available_options(),
            )
            self.xyxy_converter = ask_question(question=question, hint=hint)
        return CachableXYXYConverter.resolve(self.xyxy_converter).value

    def to_json(self) -> JSONDict:
        json_dict = {
            **super().to_json(),
            "is_label_first": self.is_label_first,
            "xyxy_converter": CachableXYXYConverter.resolve(self.xyxy_converter).name,
        }
        return json_dict

    def overwrite_missing_params(self, data: JSONDict):
        super().overwrite_missing_params(data)
        if self.is_label_first is None:
            self.is_label_first = data.get("is_label_first")
        if self.xyxy_converter is None:
            self.xyxy_converter = data.get("xyxy_converter")
