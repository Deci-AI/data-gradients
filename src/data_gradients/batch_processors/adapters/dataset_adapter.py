from typing import Callable, Union, Tuple, List, Mapping

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms

from data_gradients.batch_processors.adapters.tensor_extractor import get_tensor_extractor_options
from data_gradients.config.data.data_config import DataConfig
from data_gradients.config.data.questions import Question, text_to_yellow

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
        images = self._extract_images(data)
        labels = self._extract_labels(data)
        return self._to_torch(images), self._to_torch(labels)

    def _extract_images(self, data: SupportedData) -> torch.Tensor:
        images_extractor = self._get_images_extractor(data)
        return images_extractor(data)

    def _extract_labels(self, data: SupportedData) -> torch.Tensor:
        labels_extractor = self._get_labels_extractor(data)
        return labels_extractor(data)

    def _get_images_extractor(self, data: SupportedData) -> Callable[[SupportedData], torch.Tensor]:
        if self.data_config.images_extractor is not None:
            return self.data_config.get_images_extractor()

        # We use the heuristic that a tuple of 2 should represent (image, label) in this order
        if isinstance(data, (Tuple, List)) and len(data) == 2:
            if isinstance(data[0], (torch.Tensor, np.ndarray, PIL.Image.Image)):
                self.data_config.images_extractor = "[0]"  # We save it for later use
                return self.data_config.get_images_extractor()  # This will return a callable

        # Otherwise, we ask the user how to map data -> image
        if isinstance(data, (Tuple, List, Mapping, Tuple, List)):
            description, options = get_tensor_extractor_options(data)
            question = Question(question=f"Which tensor represents your {text_to_yellow('Image(s)')} ?", options=options)
            return self.data_config.get_images_extractor(question=question, hint=description)

        raise NotImplementedError(
            f"Got object {type(data)} from Data Iterator which is not supported!\n"
            f"Please implement a custom `images_extractor` for your dataset. "
            f"You can find more detail about this in our documentation: https://github.com/Deci-AI/data-gradients"
        )

    def _get_labels_extractor(self, data: SupportedData) -> Callable[[SupportedData], torch.Tensor]:
        if self.data_config.labels_extractor is not None:
            return self.data_config.get_labels_extractor()

        # We use the heuristic that a tuple of 2 should represent (image, label) in this order
        if isinstance(data, (Tuple, List)) and len(data) == 2:
            if isinstance(data[1], (torch.Tensor, np.ndarray, PIL.Image.Image)):
                self.data_config.labels_extractor = "[1]"  # We save it for later use
                return self.data_config.get_labels_extractor()  # This will return a callable

        # Otherwise, we ask the user how to map data -> labels
        if isinstance(data, (Tuple, List, Mapping, Tuple, List)):
            description, options = get_tensor_extractor_options(data)
            question = Question(question=f"Which tensor represents your {text_to_yellow('Label(s)')} ?", options=options)
            return self.data_config.get_labels_extractor(question=question, hint=description)

        raise NotImplementedError(
            f"Got object {type(data)} from Data Iterator which is not supported!\n"
            f"Please implement a custom `labels_extractor` for your dataset. "
            f"You can find more detail about this in our documentation: https://github.com/Deci-AI/data-gradients"
        )

    @staticmethod
    def _to_torch(tensor: Union[np.ndarray, PIL.Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor)
        elif isinstance(tensor, PIL.Image.Image):
            return transforms.ToTensor()(tensor)
        else:
            return tensor
