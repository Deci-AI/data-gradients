from abc import ABC, abstractmethod

import cv2
import numpy as np


class ImageChannels(ABC):
    def __init__(self, channels_str: str):
        self.channels_str = channels_str

    def __repr__(self):
        return self.channels_str

    def __len__(self) -> int:
        return len(self.channels_str)

    @property
    def format_idx(self):
        return self.channels_str.index(self.format_str)

    def get_channels_to_visualize(self, image: np.ndarray) -> np.ndarray:
        return image[:, :, self.format_idx : self.format_idx + len(self)]

    @staticmethod
    @abstractmethod
    def validate_channels(channels_str: str) -> bool:
        raise NotImplementedError()

    @property
    def format_str(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def convert_image_to_lab(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def compute_image_luminescence(self, image: np.ndarray) -> np.ndarray:
        return self.convert_image_to_lab(image)[0].mean()

    def to_str(self) -> str:
        return self.channels_str

    @classmethod
    def from_str(cls, channels_str: str) -> "ImageChannels":
        return image_channel_instance_factory(channels_str)


class RGBChannels(ImageChannels):
    @property
    def format_str(self) -> str:
        return "RGB"

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        return channels_str.count("RGB") == 1

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return self.get_channels_to_visualize(image)

    def convert_image_to_lab(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_RGB2LAB)


class BGRChannels(ImageChannels):
    @property
    def format_str(self) -> str:
        return "BGR"

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        return channels_str.count("BGR") == 1

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_BGR2RGB)

    def convert_image_to_lab(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_BGR2LAB)


class GrayscaleChannels(ImageChannels):
    @property
    def format_str(self) -> str:
        return "L"

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        return channels_str.count("L") == 1 and "LAB" not in channels_str

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_GRAY2RGB)

    def convert_image_to_lab(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_GRAY2RGB)  # No way to convert to LAB directly
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_RGB2LAB)

    def compute_image_luminescence(self, image: np.ndarray) -> np.ndarray:
        # We override the default implementation, because this is more computationally efficient.
        return self.get_channels_to_visualize(image).mean()


class LABChannels(ImageChannels):
    @property
    def format_str(self) -> str:
        return "LAB"

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        return channels_str.count("LAB") == 1

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_LAB2RGB)

    def convert_image_to_lab(self, image: np.ndarray) -> np.ndarray:
        return image


def image_channel_instance_factory(channels_str: str) -> ImageChannels:
    available_channel_classes = [RGBChannels, BGRChannels, GrayscaleChannels, LABChannels]

    potential_channel_classes = [ChannelClass(channels_str) for ChannelClass in available_channel_classes if ChannelClass.validate_channels(channels_str)]

    if len(potential_channel_classes) == 1:
        return potential_channel_classes[0]
    if len(potential_channel_classes) > 1:
        formats_str = ", ".join([channel_class.format_str for channel_class in potential_channel_classes])
        raise ValueError(
            f"Image channel representation `channels_str={channels_str}` is ambiguous between the following formats: {formats_str}\n"
            f"Please make sure to provide a channel representation that is not ambiguous."
        )
    else:
        raise ValueError(f"Unsupported channel format for string: {channels_str}")
