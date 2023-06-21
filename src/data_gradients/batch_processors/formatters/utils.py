import torch
from torch import Tensor

from data_gradients.batch_processors.utils import channels_last_to_first


class DatasetFormatError(Exception):
    ...


def drop_nan(tensor: Tensor) -> Tensor:
    """Remove rows containing NaN values from a given PyTorch tensor.

    :param tensor:  Tensor with shape (N, M) where N is the number of rows and M is the number of columns.
    :return:        Tensor with the same number of columns as the input tensor, but without rows containing NaN.
    """
    nans = torch.isnan(tensor)
    if nans.any():
        nan_indices = set(nans.nonzero()[:, 0].tolist())
        all_indices = set(i for i in range(tensor.shape[0]))
        valid_indices = all_indices - nan_indices
        return tensor[valid_indices]
    return tensor


def ensure_channel_first(images: Tensor, n_image_channels: int) -> Tensor:
    """Images should be [BS, C, H, W]. If [BS, W, H, C], permute

    :param images:              Tensor
    :param n_image_channels:    Number of image channels (3 for RGB, 1 for grayscale)
    :return: images:            Tensor [BS, C, H, W]
    """
    if images.shape[1] != n_image_channels and images.shape[-1] == n_image_channels:
        images = channels_last_to_first(images)
    return images


def ensure_images_shape(images: Tensor, n_image_channels: int) -> Tensor:
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
