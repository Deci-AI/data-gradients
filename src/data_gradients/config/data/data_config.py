import os
import logging
import torch
import appdirs
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Union

import data_gradients
from data_gradients.config.data.questions import Question, ask_question, text_to_yellow
from data_gradients.config.data.caching_utils import TensorExtractorResolver, XYXYConverterResolver
from data_gradients.config.data.typing import SupportedDataType, JSONDict
from data_gradients.utils.detection import XYXYConverter
from data_gradients.utils.utils import safe_json_load, write_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig(ABC):
    """Data class for handling Dataset/Dataloader configuration.

    Works as a regular dataclass, but with some additional features:
        - Getter functions that ask the user for information if this information was not provided yet.
        - Caching system, that supports saving and loading of any non-callable attribute.
            Also supports saving and loading from callable defined within DataGradients.
    """

    use_cache: bool = False  # To avoid user facing unexpected errors asking for nothing. Examples should set it to True.
    images_extractor: Union[None, str, Callable[[SupportedDataType], torch.Tensor]] = None
    labels_extractor: Union[None, str, Callable[[SupportedDataType], torch.Tensor]] = None

    DEFAULT_CACHE_DIR: str = field(default=appdirs.user_cache_dir("DataGradients", "Deci"), init=False)

    @classmethod
    def load_from_json(cls, filename: str, dir_path: Optional[str] = None) -> "DataConfig":
        """Load an instance of DataConfig directly from a cache file.
        :param filename: Name of the cache file. This should include ".json" extension.
        :param dir_path: Path to the folder where the cache file is located. By default, the cache file will be loaded from the user cache directory.
        :return: An instance of DataConfig loaded from the cache file.
        """
        dir_path = dir_path or cls.DEFAULT_CACHE_DIR
        path = os.path.join(dir_path, filename)
        try:
            return cls(**cls._load_json_dict(path=path))
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

    def write_to_json(self, filename: str, dir_path: Optional[str] = None):
        """Save the serializable representation of the class to a .json file.
        :param filename: Name of the cache file. This should include ".json" extension.
        :param dir_path: Path to the folder where the cache file is located. By default, the cache file will be loaded from the user cache directory.
        """
        dir_path = dir_path or self.DEFAULT_CACHE_DIR
        path = os.path.join(dir_path, filename)
        if not path.endswith(".json"):
            raise ValueError(f"`{path}` should end with `.json`")

        json_dict = {"metadata": {"__version__": data_gradients.__version__}, "attributes": self.to_json()}
        write_json(json_dict=json_dict, path=path)

    def to_json(self) -> JSONDict:
        """Convert the dataclass into a serializable representation that can be saved and loaded safely.
        :return: JSON like dictionary, that can be used to create a new instance of the object.
        """
        json_dict = {
            "images_extractor": TensorExtractorResolver.to_string(self.images_extractor),
            "labels_extractor": TensorExtractorResolver.to_string(self.labels_extractor),
        }
        return json_dict

    def fill_missing_params_with_cache(self, cache_filename: str, cache_dir_path: Optional[str] = None):
        """Load an instance of DataConfig directly from a cache file.
        :param cache_filename: Name of the cache file. This should include ".json" extension.
        :param cache_dir_path: Path to the folder where the cache file is located. By default, the cache file will be loaded from the user cache directory.
        :return: An instance of DataConfig loaded from the cache file.
        """
        dir_path = cache_dir_path or self.DEFAULT_CACHE_DIR
        path = os.path.join(dir_path, cache_filename)
        if self.use_cache:
            cache_dict = self._load_json_dict(path=path)
            if cache_dict:
                logger.info(
                    f"Cache activated for `{self.__class__.__name__}`. This will be used to set attributes that you did not set manually. "
                    f"Please set `use_cache=False` if you want to deactivate it."
                )
                self._fill_missing_params(json_dict=cache_dict)
        else:
            logger.info(f"Cache deactivated for `{self.__class__.__name__}`. Please set `load_cache=True` if you want to activate it.")

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
