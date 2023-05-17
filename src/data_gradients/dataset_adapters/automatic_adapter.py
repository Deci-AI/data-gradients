from typing import Iterable

import numpy as np
import torch

from data_gradients.dataset_adapters import SegmentationDatasetAdapter, SegmentationSample


class MagicSegmentationDataLoaderAdapter(SegmentationDatasetAdapter):
    """
    SegmentationDatasetAdapter implementation of the Cityscapes dataset class from torchvision.
    """

    def __init__(self, dataloader, image_key = 0, target_key = 1):
        self.dataloader = dataloader
        self.image_key = image_key
        self.target_key = target_key
        # self.class_names = # [c.name for c in dataset.classes]

    # def get_num_classes(self) -> int:
    #     return len(self.dataset.classes)
    #
    # def get_class_names(self) -> List[str]:
    #     return self.dataset.classes

    def get_iterator(self) -> Iterable[SegmentationSample]:
        sample_id = 0
        for batch in self.dataloader:
            images_batch = batch[self.image_key]
            masks_batch = batch[self.target_key]

            if torch.is_tensor(images_batch) and torch.is_tensor(masks_batch):
                images_batch = images_batch.cpu().numpy()
                masks_batch = masks_batch.cpu().numpy()

                if images_batch.ndim == 4 and masks_batch.ndim == 3 and images_batch.shape[2:] == masks_batch.shape[1:] and not torch.is_floating_point(masks_batch):
                    # Images are in NCHW format, masks are in NHW format, and masks are integer type
                    images_batch = np.transpose(images_batch, (0, 2, 3, 1)) # NHWC

                elif images_batch.ndim == 4 and masks_batch.ndim == 4 and images_batch.shape[2:] == masks_batch.shape[2:] and torch.is_floating_point(masks_batch) and masks_batch.shape[1] == 1:
                    # Images are in NCHW format, masks are in NCHW format, and masks are float and has only one channel.
                    # Most likely it's a binary segmentation mask.
                    images_batch = np.transpose(images_batch, (0, 2, 3, 1)) # NHWC
                    masks_batch = np.squeeze(masks_batch, axis=1) # NHW
                else:
                    raise RuntimeError(f"Unexpected shapes of images and masks: {images_batch.shape} and {masks_batch.shape}")

                for image, mask in zip(images_batch, masks_batch):
                    yield SegmentationSample(str(sample_id), image, mask)
                    sample_id += 1

            else:
                raise RuntimeError(f"Unexpected types of images and masks: {type(images_batch)} and {type(masks_batch)}")
