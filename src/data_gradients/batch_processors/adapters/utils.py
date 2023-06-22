from typing import Union

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms


def to_torch(tensor: Union[np.ndarray, PIL.Image.Image, torch.Tensor]) -> torch.Tensor:
    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    elif isinstance(tensor, PIL.Image.Image):
        return transforms.ToTensor()(tensor)
    else:
        return tensor
