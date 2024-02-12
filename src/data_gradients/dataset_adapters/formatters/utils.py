import logging

from data_gradients.dataset_adapters.utils import channels_last_to_first


from abc import ABC, abstractmethod
from typing import Optional, List, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


class DatasetFormatError(Exception):
    ...


def ensure_channel_first(images: Union[torch.Tensor, np.ndarray], n_image_channels: int) -> Union[torch.Tensor, np.ndarray]:
    """Images should be [BS, C, H, W]. If [BS, W, H, C], permute

    :param images:              Tensor
    :param n_image_channels:    Number of image channels (3 for RGB, 1 for grayscale)
    :return: images:            Tensor [BS, C, H, W]
    """
    if images.shape[1] != n_image_channels and images.shape[-1] == n_image_channels:
        images = channels_last_to_first(images)
    return images


def check_images_shape(images: Union[torch.Tensor, np.ndarray], n_image_channels: int) -> Union[torch.Tensor, np.ndarray]:
    """Validate images dimensions are (BS, C, H, W)

    :param images:              Tensor [BS, C, H, W]
    :param n_image_channels:    Number of image channels (C = 3 for RGB, C = 1 for grayscale)
    :return: images:            Tensor [BS, C, H, W]
    """
    if images.dim() != 4:
        raise ValueError(f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}")

    if images.shape[1] != n_image_channels:
        raise ValueError(f"Images should have {n_image_channels} number of channels. Got {min(images[0].shape)}")

    return images


class ImageFormat(ABC):
    @abstractmethod
    def convert_image_to_float(self, images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def convert_image_from_float(self, images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def to_json(self) -> dict:
        """Serializes the normalizer to JSON."""
        pass

    @staticmethod
    @abstractmethod
    def from_json(json_data: dict) -> Optional["ImageFormat"]:
        """Deserializes the normalizer from JSON."""
        pass


class FloatImageFormat(ImageFormat):
    def convert_image_to_float(self, images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return images

    def convert_image_from_float(self, images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return images

    def to_json(self) -> dict:
        return {"type": "FloatImageFormat", "description": "The images in your dataset have values set between [0-1]."}

    @staticmethod
    def from_json(json_data: dict) -> "FloatImageFormat":
        return FloatImageFormat()


class Uint8ImageFormat(ImageFormat):
    def convert_image_to_float(self, images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return images / 255.0

    def convert_image_from_float(self, images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if torch.is_tensor(images):
            return (images * 255).to(torch.uint8)
        else:
            return (images * 255).astype(np.uint8)

    def to_json(self) -> dict:
        return {"type": "Uint8ImageFormat", "description": "The images in your dataset have values set between [0-255]."}

    @staticmethod
    def from_json(json_data: dict) -> "Uint8ImageFormat":
        return Uint8ImageFormat()


class ScaledFloatImageFormat(ImageFormat):
    def __init__(self, mean: List[float], std: List[float]):
        self.np_mean = np.array(mean)
        self.np_std = np.array(std)

        self.torch_mean = torch.from_numpy(self.np_mean).view(3, 1, 1)
        self.torch_std = torch.from_numpy(self.np_std).view(3, 1, 1)

    def convert_image_to_float(self, images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if torch.is_tensor(images):
            return (images - self.torch_mean) / self.torch_std
        else:
            return (images - self.np_mean) / self.np_std

    def convert_image_from_float(self, images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if torch.is_tensor(images):
            return (images * self.torch_std) + self.torch_mean
        else:
            return (images * self.np_std) + self.np_mean

    def to_json(self) -> dict:
        return {
            "type": "ScaledFloatImageFormat",
            "description": "The images in your dataset have values set between [-1, 1].",
            "data": {"mean": self.np_mean.tolist(), "std": self.np_std.tolist()},
        }

    @classmethod
    def from_json(cls, json_data: dict) -> Optional["ScaledFloatImageFormat"]:
        if "data" not in json_data or "mean" not in json_data["data"] or "std" not in json_data["data"]:
            logger.warn(
                "It looks like the cache of your image normalizer does not include any information about the mean and standard deviation."
                "This may be due to invalid cache format, and therefore normalizer will not be loaded from cache."
            )
            return None
        return cls(mean=json_data["data"]["mean"], std=json_data["data"]["std"])


class ImageFormatFactory:

    _normalizers = {
        Uint8ImageFormat.__name__: Uint8ImageFormat,
        FloatImageFormat.__name__: FloatImageFormat,
        ScaledFloatImageFormat.__name__: ScaledFloatImageFormat,
    }

    @staticmethod
    def get_normalizer_from_cache(json_data: dict) -> Optional[ImageFormat]:
        image_format_type = json_data.get("type")
        image_format_class = ImageFormatFactory._normalizers.get(image_format_type)
        if image_format_class:
            return image_format_class.from_json(json_data)
        return None
