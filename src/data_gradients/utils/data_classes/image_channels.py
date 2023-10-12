from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np


class ImageChannels(ABC):
    """Represent the channels of an image.

    >> ImageChannels.from_str("LAB") # This represents the format of LAB color space.
    >> ImageChannels.from_str("RGBO") # This represents the format of an image with (R, G, B, ?) channels, where `?` can be any other channel (Depth, ...)
    """

    def __init__(self, channels_str: str, idx_to_visualize: slice, channel_names: List[str]):
        """
        :param channels_str: The string representation of the channels. `RGB`, `LAB`, `RGBO`, ... Each channel is represented by a letter.
        """
        if not self.validate_channels(channels_str=channels_str):
            raise ValueError(f"Invalid `channels_str={channels_str}` for `{self.__class__.__name__}`")
        self.channels_str = channels_str
        self.idx_to_visualize = idx_to_visualize
        self.channel_names = channel_names

    def __repr__(self):
        return self.channels_str

    def __len__(self) -> int:
        return len(self.channels_str)

    @staticmethod
    @abstractmethod
    def validate_channels(channels_str: str) -> bool:
        raise NotImplementedError()

    def get_channels_to_visualize(self, image: np.ndarray) -> np.ndarray:
        """Get the channels that represent an image
        :param image:   The image to convert of shape [H, W, C] with C the number of channels of the image.
        :return:        The converted image in shape [H, W, n] with n the number of channels to visualize.
        """
        return image[:, :, self.idx_to_visualize]

    @abstractmethod
    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert an input image to RGB.

        :param image:   The image to convert of shape [H, W, C].
                        This image should include the channel as defined by `self.channels_str`.
        :return:        The converted image in shape [H, W, 3] with channels RGB
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_mean_image_intensity(self, image: np.ndarray) -> float:
        """Convert the mean intensity of an image.

        :param image:   The image to convert of shape [H, W, C].
                        This image should include the channel as defined by `self.channels_str`.
        :return:        float, representing the mean intensity of the input image.
        """
        raise NotImplementedError()

    def to_str(self) -> str:
        """String representation of the image channels.

        This should be reverse `from_str`:
            >> assert ImageChannels.from_str(channel_str).to_str() == channel_str
        """
        return self.channels_str

    @classmethod
    def from_str(cls, channels_str: str) -> "ImageChannels":
        """
        Create an ImageChannels instance based on the channel representation string.

        The function determines the appropriate subclass of ImageChannels (e.g., RGBChannels, BGRChannels) based on the
        given string representation of the channels. It also considers additional channels (represented by 'O') present
        in the image.

        :param channels_str:    String representing the channels in the image. Valid channel representations include:
                                    - 'RGB': Red, Green, Blue channels
                                    - 'BGR': Blue, Green, Red channels
                                    - 'G': Grayscale channel
                                    - 'LAB': Luminance, A and B color channels
                                    - 'O': Any additional channel (e.g., Depth)
        :return:                ImageChannels instance corresponding to the channel representation string.

        Examples:
        >> image_channel_instance_factory('RGB')
        RGB
        >> image_channel_instance_factory('RGBO')
        RGBO (with the last channel being an additional channel, e.g., Depth)

        Notes:
        - The order of channels in the string representation is significant and should match the order of channels in the image.
        """
        return image_channel_instance_factory(channels_str)


class RGBChannels(ImageChannels):
    def __init__(self, channels_str: str):
        format_idx = channels_str.index("RGB")

        channel_names = []
        for i, channel_str in enumerate(channels_str):
            if channel_str == "R":
                channel_names.append("Red")
            elif channel_str == "G":
                channel_names.append("Green")
            elif channel_str == "B":
                channel_names.append("Blue")
            else:
                channel_names.append(f"Channel_{i}")

        super().__init__(
            channels_str=channels_str,
            idx_to_visualize=slice(format_idx, format_idx + 3),
            channel_names=channel_names,
        )

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        """A valid RGB channel representation string should have `RGB` once and `O` the rest of the channels.`"""
        return channels_str.count("RGB") == 1 and channels_str.count("O") == len(channels_str) - 3

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return self.get_channels_to_visualize(image)

    def compute_mean_image_intensity(self, image: np.ndarray) -> float:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_RGB2LAB)[0].mean()


class BGRChannels(ImageChannels):
    def __init__(self, channels_str: str):
        format_idx = channels_str.index("BGR")

        channel_names = []
        for i, channel_str in enumerate(channels_str):
            if channel_str == "R":
                channel_names.append("Red")
            elif channel_str == "G":
                channel_names.append("Green")
            elif channel_str == "B":
                channel_names.append("Blue")
            else:
                channel_names.append(f"Channel_{i}")

        super().__init__(
            channels_str=channels_str,
            idx_to_visualize=slice(format_idx, format_idx + 3),
            channel_names=channel_names,
        )

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        """A valid BGR channel representation string should have `BGR` once and `O` the rest of the channels.`"""
        return channels_str.count("BGR") == 1 and channels_str.count("O") == len(channels_str) - 3

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_BGR2RGB)

    def compute_mean_image_intensity(self, image: np.ndarray) -> float:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_BGR2LAB)[0].mean()


class GrayscaleChannels(ImageChannels):
    def __init__(self, channels_str: str):
        format_idx = channels_str.index("G")

        channel_names = []
        for i, channel_str in enumerate(channels_str):
            if channel_str == "G":
                channel_names.append("Grayscale")
            else:
                channel_names.append(f"Channel_{i}")

        super().__init__(
            channels_str=channels_str,
            idx_to_visualize=slice(format_idx, format_idx + 3),
            channel_names=channel_names,
        )

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        """A valid Grayscale channel representation string should have `G` once and `O` the rest of the channels.`"""
        return channels_str.count("G") == 1 and channels_str.count("O") == len(channels_str) - 1

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_GRAY2RGB)

    def compute_mean_image_intensity(self, image: np.ndarray) -> float:
        return image.mean()


class LABChannels(ImageChannels):
    def __init__(self, channels_str: str):
        format_idx = channels_str.index("LAB")

        channel_names = []
        for i, channel_str in enumerate(channels_str):
            if channel_str == "L":
                channel_names.append("Luminance")
            elif channel_str == "A":
                channel_names.append("A")
            elif channel_str == "B":
                channel_names.append("B")
            else:
                channel_names.append(f"Channel_{i}")

        super().__init__(
            channels_str=channels_str,
            idx_to_visualize=slice(format_idx, format_idx + 3),
            channel_names=channel_names,
        )

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        """A valid LAB channel representation string should have `LAB` once and `O` the rest of the channels.`"""
        return channels_str.count("LAB") == 1 and channels_str.count("O") == len(channels_str) - 3

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.get_channels_to_visualize(image), cv2.COLOR_LAB2RGB)

    def compute_mean_image_intensity(self, image: np.ndarray) -> float:
        return image[0].mean()


class OtherChannels(ImageChannels):
    def __init__(self, channels_str: str):
        super().__init__(
            channels_str=channels_str,
            idx_to_visualize=slice(0, 1),
            channel_names=[f"Channel_{i}" for i in range(len(channels_str))],
        )

    @staticmethod
    def validate_channels(channels_str: str) -> bool:
        return channels_str.count("O") == len(channels_str)

    def convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return None

    def compute_mean_image_intensity(self, image: np.ndarray) -> float:
        return image.mean()


def image_channel_instance_factory(channels_str: str) -> ImageChannels:
    """
    Create an ImageChannels instance based on the channel representation string.

    The function determines the appropriate subclass of ImageChannels (e.g., RGBChannels, BGRChannels) based on the
    given string representation of the channels. It also considers additional channels (represented by 'O') present
    in the image.

    :param channels_str:    String representing the channels in the image. Valid channel representations include:
                                - 'RGB': Red, Green, Blue channels
                                - 'BGR': Blue, Green, Red channels
                                - 'G': Grayscale channel
                                - 'LAB': Luminance, A and B color channels
                                - 'O': Any additional channel (e.g., Depth)
    :return:                ImageChannels instance corresponding to the channel representation string.

    Examples:
    >> image_channel_instance_factory('RGB')
    RGB
    >> image_channel_instance_factory('RGBO')
    RGBO (with the last channel being an additional channel, e.g., Depth)

    Notes:
    - The order of channels in the string representation is significant and should match the order of channels in the image.
    """

    # Check which channel representation fits the given string.
    potential_channel_classes = [
        ChannelClass(channels_str)
        for ChannelClass in [RGBChannels, BGRChannels, GrayscaleChannels, LABChannels, OtherChannels]
        if ChannelClass.validate_channels(channels_str)
    ]

    # Only accept case without ambiguous channel representation.
    if len(potential_channel_classes) == 1:
        return potential_channel_classes[0]
    if len(potential_channel_classes) > 1:
        formats_str = ", ".join([channel_class._format_str for channel_class in potential_channel_classes])
        raise ValueError(
            f"Image channel representation `channels_str={channels_str}` is ambiguous between the following formats: {formats_str}\n"
            f"Please make sure to provide a channel representation that is not ambiguous."
        )
    else:
        raise ValueError(f"Unsupported channel format for string: {channels_str}")
