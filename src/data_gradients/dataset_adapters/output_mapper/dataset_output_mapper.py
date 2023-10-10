from typing import Callable, Union, Tuple, List, Mapping, Sequence

import PIL
import numpy as np
import torch
from omegaconf import ListConfig

from data_gradients.dataset_adapters.output_mapper.tensor_extractor import get_tensor_extractor_options
from data_gradients.dataset_adapters.config.data_config import DataConfig
from data_gradients.dataset_adapters.config.questions import FixedOptionsQuestion, text_to_yellow

SupportedData = Union[Tuple, List, Mapping, Tuple, List]


class DatasetOutputMapper:
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
        images = self._extract_images_as_tensor(data)
        labels = self._extract_labels_as_tensor(data)
        return images, labels

    def _extract_images_as_tensor(self, data: SupportedData) -> torch.Tensor:
        images_extractor = self._get_images_extractor(data)
        images = images_extractor(data)
        try:
            return self._to_torch(images)
        except TypeError:
            raise TypeError(f"{type(images)} is not a supported format for images!")
        except Exception as e:
            raise RuntimeError("Error while loading images!") from e  # Here we want the traceback

    def _extract_labels_as_tensor(self, data: SupportedData) -> torch.Tensor:
        labels_extractor = self._get_labels_extractor(data)
        labels = labels_extractor(data)
        try:
            return self._to_torch(labels)
        except TypeError:
            raise TypeError(f"{type(labels)} is not a supported format for labels!")
        except Exception as e:
            raise RuntimeError("Error while loading labels!") from e  # Here we want the traceback

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
            question = FixedOptionsQuestion(question=f"Which tensor represents your {text_to_yellow('Image(s)')} ?", options=options)
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
            question = FixedOptionsQuestion(question=f"Which tensor represents your {text_to_yellow('Label(s)')} ?", options=options)
            return self.data_config.get_labels_extractor(question=question, hint=description)

        raise NotImplementedError(
            f"Got object {type(data)} from Data Iterator which is not supported!\n"
            f"Please implement a custom `labels_extractor` for your dataset. "
            f"You can find more detail about this in our documentation: https://github.com/Deci-AI/data-gradients"
        )

    @classmethod
    def _to_torch(cls, data: Union[np.ndarray, PIL.Image.Image, torch.Tensor, str, Sequence[Union[PIL.Image.Image, np.ndarray, str]]]) -> torch.Tensor:
        """Convert various input types to a PyTorch tensor.

        :param data: Input data to be converted. This can be:
            - PyTorch tensor (in which case the function simply returns it)
            - numpy ndarray
            - PIL Image
            - string representing the path to an image file
            - list containing any combination of the above three types

        :return: The input data converted (or simply returned if already) as a PyTorch tensor.
        """
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, PIL.Image.Image):
            return torch.from_numpy(np.array(data))
        elif isinstance(data, str):
            with PIL.Image.open(data) as img:
                return torch.from_numpy(np.array(img))
        elif np.isscalar(data):
            return torch.tensor(data)
        elif isinstance(data, (list, tuple, ListConfig)):
            tensors = [cls._to_torch(t) for t in data]

            # Check if all tensors can be stacked
            reference_shape = tensors[0].shape
            for idx, t in enumerate(tensors[1:], 1):  # Start from the second tensor
                if t.shape != reference_shape:
                    raise RuntimeError(
                        f"Error while attempting to stack tensors. Tensor at index 0 has shape {reference_shape} "
                        f"while tensor at index {idx} has shape {t.shape}. All tensors must have the same shape to be stacked."
                    )
            return torch.stack(tensors)
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")
