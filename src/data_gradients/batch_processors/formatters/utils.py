import torch
from torch import Tensor

from data_gradients.batch_processors.utils import channels_last_to_first


def ensure_images_shape(images: Tensor, n_image_channels: int) -> Tensor:
    """
    Validating images dimensions are (BS, Channels, W, H)
    :param images: Tensor [BS, C, W, H]
    :return: images: Tensor [BS, C, W, H]
    """
    if images.dim() != 4:
        raise ValueError(f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}")

    if images.shape[1] != n_image_channels and images.shape[-1] != n_image_channels:
        raise ValueError(f"Images should have {n_image_channels} number of channels. Got {min(images[0].shape)}")

    return images


def ensure_channel_first(images: Tensor, n_image_channels: int) -> Tensor:
    """Images should be [BS, C, W, H]. If [BS, W, H, C], permute

    :param images: Tensor
    :return: images: Tensor [BS, C, W, H]
    """
    if images.shape[1] != n_image_channels and images.shape[-1] == n_image_channels:
        images = channels_last_to_first(images)
    return images


def drop_nan(tensor: Tensor) -> Tensor:
    nans = torch.isnan(tensor)
    if nans.any():
        nan_indices = set(nans.nonzero()[:, 0].tolist())
        all_indices = set(i for i in range(tensor.shape[0]))
        valid_indices = all_indices - nan_indices
        return tensor[valid_indices]
    return tensor
