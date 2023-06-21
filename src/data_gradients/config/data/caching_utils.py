from dataclasses import dataclass
from typing import Optional, Any, Union, Callable

import torch

from data_gradients.batch_processors.adapters.tensor_extractor import NestedDataLookup
from data_gradients.config.data.typing import SupportedDataType
from data_gradients.utils.detection import XYXYConverter

# This is used as a prefix to recognize parameters that are not cachable.
NON_CACHABLE_PREFIX = "[Non-cachable]"


class CacheLoadingError(Exception):
    def __init__(self, key: str, value: str):
        self.key, self.value = key, value
        message = (
            f"Error while trying to load attribute `{key}` from cache... (with value `{value}`).\n"
            f"It seems that this object was passed to the `DataConfig` in the previous run, but not this time.\n"
            f"Please:\n"
            f"     - Either pass the same `{key}` to the `DataConfig`.\n"
            f"     - Or disable loading config from cache when instantiating `DataConfig`.\n"
        )
        super().__init__(message)


@dataclass
class CachableParam:
    """Dataclass representing a parameter that can be cached.
    This combines the value of any parameter, as used in the code, and the name of the parameter that will be used in the cache.

    :attr value:    The value of the parameter, will be used in the code.
    :attr name:     The name of the parameter, will be used to load/save cache.
    """

    value: Optional[Any]
    name: Optional[str]


class TensorExtractorResolver:
    @staticmethod
    def to_callable(tensor_extractor: Union[str, Callable[[SupportedDataType], torch.Tensor]]) -> Callable[[SupportedDataType], torch.Tensor]:
        """Ensures the input `tensor_extractor` to be a callable.

        For example:
            >> TensorExtractorResolver.to_callable("[1].bbox")
            # NestedDataLookup("[1].bbox")

            >> TensorExtractorResolver.to_callable(lambda x: x[1]["bbox"])
            # lambda x: x[1]["bbox"]

        :param tensor_extractor: Either a string representation (e.g. `[1].bbox`) or a custom callable (e.g. lambda x: x[1]["bbox"])
        :return: Tensor extractor, extracting a specific tensor from the dataset outputs.
        """
        return TensorExtractorResolver._resolve(tensor_extractor).value

    @staticmethod
    def to_string(tensor_extractor: Union[None, str, Callable[[SupportedDataType], torch.Tensor]]) -> str:
        """Ensures the input `tensor_extractor` to be a string.

        For example:
            >> TensorExtractorResolver.to_string(None)
            # "None"

            >> TensorExtractorResolver.to_string("[1].bbox")
            # "[1].bbox"

            >> TensorExtractorResolver.to_string(lambda x: x[1]["bbox"])
            # [Non-cachable] - function <lambda> at 0x102ca58b0

        :param tensor_extractor: Either None, a string representation (e.g. `[1].bbox`) or a custom callable (e.g. lambda x: x[1]["bbox"])
        :return: String representing this function, to support loading/saving this function into cache.
        """
        return TensorExtractorResolver._resolve(tensor_extractor).name

    @staticmethod
    def _resolve(tensor_extractor: Union[None, str, Callable[[torch.Tensor], torch.Tensor]]) -> CachableParam:
        """Translate the input `tensor_extractor` into both:
            - value: Callable that extract a specific tensor (e.g. Image(s) or Label(s)) form a dataset output, which will be used in the code.
            - name:  String representing this function, to support loading/saving this function into cache.

        This provides a unique interface to support both callables and strings.

        For example:
            >> TensorExtractorResolver.resolve("[1].bbox")
            # CachableParam(value=NestedDataLookup("[1].bbox"), name="[1].bbox")

            >> TensorExtractorResolver.resolve(lambda x: x[1]["bbox"])
            # CachableParam(value=lambda x: x[1]["bbox"], name="[Non-cachable] - function <lambda> at 0x102ca58b0")

        :param tensor_extractor: Either None, a string representation (e.g. `[1].bbox`) or a custom callable (e.g. lambda x: x[1]["bbox"])
        :return: Dataclass including both the value (used in the code) and the name (used in the cache) of this function.
        """
        if tensor_extractor is None:
            return CachableParam(value=None, name="None")

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


class XYXYConverterResolver:
    @staticmethod
    def to_callable(xyxy_converter: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> Callable[[torch.Tensor], torch.Tensor]:
        """Ensures the input `xyxy_converter` to be a callable.

        For example:
            >> XYXYConverterResolver.to_callable("xywh")
            # XYXYConverter("xywh")

            >> XYXYConverterResolver.to_callable(custom_xywh2xyxy)
            # custom_xywh2xyxy

        :param xyxy_converter: Either a string representation (e.g. `xywh`) or a custom callable (e.g. custom_xywh2xyxy or lambda bbox: ...)
        :return: Callable converting bboxes into xyxy.
        """
        return XYXYConverterResolver._resolve(xyxy_converter).value

    @staticmethod
    def to_string(xyxy_converter: Union[None, str, Callable[[torch.Tensor], torch.Tensor]]) -> str:
        """Ensures the input `xyxy_converter` to be a string.

        For example:
            >> XYXYConverterResolver.to_string(None)
            # "None"

            >> XYXYConverterResolver.to_string("xywh")
            # "xyxy"

            >> XYXYConverterResolver.to_string(custom_xywh2xyxy)
            # [Non-cachable] - <function custom_xywh2xyxy at 0x102ca5820>

        :param xyxy_converter: Either None, a string representation (e.g. `xywh`) or a custom callable (e.g. custom_xywh2xyxy or lambda bbox: ...)
        :return: String representing this function, to support loading/saving this function into cache.
        """
        return XYXYConverterResolver._resolve(xyxy_converter).name

    @staticmethod
    def _resolve(xyxy_converter: Union[None, str, Callable[[torch.Tensor], torch.Tensor]]) -> CachableParam:
        """Translate the input `xyxy_converter` into both:
            - value: Callable that converts bboxes into xyxy, which will be used in the code.
            - name:  String representing this function, to support loading/saving this function into cache.

        For example:
            >> XYXYConverterResolver.resolve("xywh")
            # CachableParam(value=XYXYConverter("xywh"), name="xyxy")

            >> XYXYConverterResolver.resolve(custom_xywh2xyxy)
            # CachableParam(value=custom_xywh2xyxy, name="[Non-cachable] - <function custom_xywh2xyxy at 0x102ca5820>")

        :param xyxy_converter: Either None, a string representation (e.g. `xywh`) or a custom callable (e.g. custom_xywh2xyxy or lambda bbox: ...)
        :return: Dataclass including both the value (used in the code) and the name (used in the cache) of this function.
        """
        if xyxy_converter is None:
            return CachableParam(value=None, name="None")

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
