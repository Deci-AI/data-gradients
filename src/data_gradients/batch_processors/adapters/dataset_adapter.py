from typing import Callable, Union, Tuple, List, Mapping

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms

from data_gradients.batch_processors.adapters.tensor_extractor import get_tensor_extractor_options
from data_gradients.config.data_config import DataConfig, Question

SupportedData = Union[Tuple, List, Mapping, Tuple, List]


class DatasetAdapter:
    """Class responsible to convert raw batch (coming from dataloader) into a batch of image and a batch of labels."""

    def __init__(self, data_config: DataConfig):
        self.data_config = data_config

    def extract(self, data: SupportedData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw batch (coming from dataloader) into a batch of image and a batch of labels.

        :param data: Raw batch (coming from dataloader without any modification).
        :return:
            - images: Batch of images
            - labels: Batch of labels
        """
        images_extractor = self._get_images_extractor(data)
        labels_extractor = self._get_labels_extractor(data)
        images, labels = images_extractor(data), labels_extractor(data)
        return self.to_torch(images), self.to_torch(labels)

    def _get_images_extractor(self, data: SupportedData) -> Callable[[SupportedData], torch.Tensor]:
        if self.data_config.images_extractor is not None:
            return self.data_config.get_images_extractor()

        # If data == Tuple[Union[Tensor, np.ndarray, Image], ...]
        if isinstance(data, (Tuple, List)) and len(data) == 2:
            if isinstance(data[0], (torch.Tensor, np.ndarray, PIL.Image.Image)):
                self.data_config.images_extractor = "[0]"
                return self.data_config.get_images_extractor()

        # If data != data == Tuple[Union[Tensor, np.ndarray, Image], ...] but we can still extract the image
        if isinstance(data, (Tuple, List, Mapping, Tuple, List)):
            options = get_tensor_extractor_options(data)
            question = Question(question="Which tensor represents your Images ?", options=options)
            return self.data_config.get_images_extractor(question=question)

        raise NotImplementedError(f"Got object {type(data)} from Data Iterator - supporting (Tuple, List, Mapping, Tuple, List) only!")

    def _get_labels_extractor(self, data: SupportedData) -> Callable[[SupportedData], torch.Tensor]:
        # If data == Tuple[Union[Tensor, np.ndarray, Image], ...]
        if isinstance(data, (Tuple, List)) and len(data) == 2:
            if isinstance(data[1], (torch.Tensor, np.ndarray, PIL.Image.Image)):
                self.data_config.labels_extractor = "[1]"
                return self.data_config.get_labels_extractor()

        # If data != data == Tuple[Union[Tensor, np.ndarray, Image], ...] but we can still extract the image
        if isinstance(data, (Tuple, List, Mapping, Tuple, List)):
            options = get_tensor_extractor_options(data)
            question = Question(question="Which tensor represents your Labels ?", options=options)
            return self.data_config.get_labels_extractor(question=question)

        raise NotImplementedError(f"Got object {type(data)} from Data Iterator - supporting (Tuple, List, Mapping, Tuple, List) only!")

    @staticmethod
    def to_torch(tensor: Union[np.ndarray, PIL.Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor)
        elif isinstance(tensor, PIL.Image.Image):
            return transforms.ToTensor()(tensor)
        else:
            return tensor
