import os
import logging

import platformdirs
import torch
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Union

import data_gradients
from data_gradients.config.data.questions import Question, ask_question, text_to_yellow
from data_gradients.config.data.caching_utils import TensorExtractorResolver, XYXYConverterResolver
from data_gradients.config.data.typing import SupportedDataType, JSONDict
from data_gradients.utils.detection import XYXYConverter
from data_gradients.utils.utils import safe_json_load, write_json

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

    cache_path: Optional[str] = None

    def __post_init__(self):
        # Once the object is initialized, we check if the cache is activated or not.
        if self.cache_path is not None and os.path.isfile(self.cache_path):
            self.update_from_cache_file()
        else:
            logger.info(f"Cache deactivated for `{self.__class__.__name__}`.")

    def update_from_cache_file(self):
        """Update the values that are not set yet, using the cache file."""
        if self.cache_path is None:
            print(f"Cache deactivated for `{self.__class__.__name__}`. To activate the cache, please set `cache_path`.")
        elif not os.path.isfile(self.cache_path):
            logger.warning(f"Expected cache file at `cache_path={self.cache_path}` but none was found. Ensure the correct path is set.")
        else:
            print(f"`{self.__class__.__name__}` will load cache from `cache_path={self.cache_path}`.")
            self._fill_missing_params_with_cache(self.cache_path)

    def dump_cache_file(self):
        """Save the current state to the cache file."""
        if self.cache_path is None:
            logger.warning(f"`{self.__class__.__name__}` cache is not saved because `cache_path={self.cache_path}` is not set.")
        else:
            self.write_to_json(self.cache_path)
            print(f"Successfully saved `{self.__class__.__name__}` cache to `cache_path={self.cache_path}`.")

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
        }
        return json_dict

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

    def get_images_extractor(self, question: Optional[Question] = None, hint: str = "") -> Callable[[SupportedDataType], torch.Tensor]:
        if self.images_extractor is None:
            self.images_extractor = ask_question(question=question, hint=hint)
        return TensorExtractorResolver.to_callable(tensor_extractor=self.images_extractor)

    def get_labels_extractor(self, question: Optional[Question] = None, hint: str = "") -> Callable[[SupportedDataType], torch.Tensor]:
        if self.labels_extractor is None:
            self.labels_extractor = ask_question(question=question, hint=hint)
        return TensorExtractorResolver.to_callable(tensor_extractor=self.labels_extractor)

    def close(self):
        """Run any action required to cleanly close the object. May include saving cache."""
        if self.cache_path is not None:
            self.dump_cache_file()


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
            question = Question(
                question=f"{text_to_yellow('Which comes first')} in your annotations, the class id or the bounding box?",
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
                question=f"What is the {text_to_yellow('bounding box format')}?",
                options=XYXYConverter.get_available_options(),
            )
            self.xyxy_converter = ask_question(question=question, hint=hint)
        return XYXYConverterResolver.to_callable(self.xyxy_converter)
