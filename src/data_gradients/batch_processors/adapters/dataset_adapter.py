from typing import Optional, Callable, Union, Tuple, List, Mapping, Any

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms

from data_gradients.batch_processors.adapters.tensor_extractor import TensorExtractor


class ImageExtractorError(Exception):
    def __init__(self):
        msg = (
            "\n\nERROR: Something went wrong when extracting Images!\n\n"
            "Please implement and pass to the config the following function\n"
            "images_extractor(data: Any) -> torch.Tensor\n\n"
            "- `data` being the output of the dataset/dataloader that you provided.\n"
            "- The function should return a Tensor representing your image(s). One of:\n"
            "  - `(BS, C, H, W)`, `(BS, H, W, C)`, `(BS, H, W)` for batch\n"
            "  - `(C, H, W)`, `(H, W, C)`, `(H, W)` for single image\n"
            "    - With `C`: number of channels (3 for RGB)\n"
        )
        super().__init__(msg)


class LabelsExtractorError(Exception):
    def __init__(self):
        msg = (
            "\n\nERROR: Something went wrong when extracting Labels!\n\n"
            "Please implement and pass to the config the following function:\n"
            "labels_extractor(data: Any) -> torch.Tensor\n\n"
            "- `data` being the output of the dataset/dataloader that you provided.\n"
            "- The function should return a Tensor representing your labels(s):\n"
            "  - For **Segmentation**, one of:\n"
            "    - `(BS, C, H, W)`, `(BS, H, W, C)`, `(BS, H, W)` for batch\n"
            "    - `(C, H, W)`, `(H, W, C)`, `(H, W)` for single image\n"
            "      - `BS`: Batch Size\n"
            "      - `C`: number of channels - 3 for RGB\n"
            "      - `H`, `W`: Height and Width\n"
            "  - For **Detection**, one of:\n"
            "    - `(BS, N, 5)`, `(N, 6)` for batch\n"
            "    - `(N, 5)` for single image\n"
            "      - `BS`: Batch Size\n"
            "      - `N`: Padding size\n"
            "      - The last dimension should include your `class_id` and `bbox` - `class_id, x, y, x, y` for instance\n"
        )
        super().__init__(msg)


class DatasetAdapter:
    """Class responsible to convert raw batch (coming from dataloader) into a batch of image and a batch of labels."""

    def __init__(self, images_extractor: Optional[Callable] = None, labels_extractor: Optional[Callable] = None):
        """
        :param images_extractor:    (Optional) function that takes the dataloader output and extract the images.
                                    If None, the user will need to input it manually in a following prompt.
        :param labels_extractor:    (Optional) function that takes the dataloader output and extract the labels.
                                    If None, the user will need to input it manually in a following prompt.
        """
        self._tensor_extractor = {0: images_extractor, 1: labels_extractor}

    def extract(self, objs: Union[Tuple, List, Mapping]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw batch (coming from dataloader) into a batch of image and a batch of labels.

        :param objs: Raw batch (coming from dataloader without any modification).
        :return:
            - images: Batch of images
            - labels: Batch of labels
        """
        if isinstance(objs, (Tuple, List)) and len(objs) == 2:
            images = objs[0] if isinstance(objs[0], torch.Tensor) else self._to_tensor(objs[0], tuple_idx=0)
            labels = objs[1] if isinstance(objs[1], torch.Tensor) else self._to_tensor(objs[1], tuple_idx=1)
        elif isinstance(objs, (Mapping, Tuple, List)):
            images = self._extract_tensor_from_container(objs, 0)
            labels = self._extract_tensor_from_container(objs, 1)
        else:
            raise NotImplementedError(f"Got object {type(objs)} from Iterator - supporting dict, tuples and lists Only!")
        return images, labels

    def _to_tensor(self, objs: Union[np.ndarray, PIL.Image.Image, Mapping], tuple_idx: int) -> torch.Tensor:
        if isinstance(objs, np.ndarray):
            return torch.from_numpy(objs)
        elif isinstance(objs, PIL.Image.Image):
            return transforms.ToTensor()(objs)
        else:
            return self._extract_tensor_from_container(objs=objs, tuple_idx=tuple_idx)

    def _extract_tensor_from_container(self, objs: Any, tuple_idx: int) -> torch.Tensor:
        mapping_fn = self._get_tensor_extractor(tuple_idx=tuple_idx, objs=objs)
        return mapping_fn(objs)

    def _get_tensor_extractor(self, objs: Any, tuple_idx: int) -> Union[Callable, TensorExtractor]:
        if self._tensor_extractor[tuple_idx] is None:
            self._tensor_extractor[tuple_idx] = TensorExtractor(objs=objs, name="image(s)" if (tuple_idx == 0) else "label(s)")
        return self._tensor_extractor[tuple_idx]

    @property
    def images_route(self) -> List[str]:
        """Represent the path (route) to extract the images from the raw batch (coming from dataloader)."""
        tensor_finder = self._tensor_extractor[0]
        return tensor_finder.path_to_tensor if isinstance(tensor_finder, TensorExtractor) else []

    @property
    def labels_route(self) -> List[str]:
        """Represent the path (route) to extract the labels from the raw batch (coming from dataloader)."""
        tensor_finder = self._tensor_extractor[1]
        return tensor_finder.path_to_tensor if isinstance(tensor_finder, TensorExtractor) else []
