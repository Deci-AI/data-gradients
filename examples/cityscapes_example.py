from typing import Any, Tuple

import cv2
from fire import Fire
from torchvision.datasets.cityscapes import Cityscapes

from data_gradients.dataset_adapters import TorchvisionCityscapesSegmentationAdapter
from data_gradients.managers.image_analysis_manager import ImageAnalysisManager
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager


class OptimizedCityscapes(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = cv2.imread(self.images[index])[..., ::-1]

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = cv2.imread(self.targets[index][i], cv2.IMREAD_GRAYSCALE)

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def main(root="e:/cityscapes"):
    train_dataset = OptimizedCityscapes(
        root=root,
        split="train",
        target_type="semantic",
    )
    val_dataset = OptimizedCityscapes(
        root=root,
        split="val",
        target_type="semantic",
    )

    dataset = {
        "train": TorchvisionCityscapesSegmentationAdapter(train_dataset),
        "val": TorchvisionCityscapesSegmentationAdapter(val_dataset),
    }

    # ImageAnalysisManager.extract_features_from_splits(dataset, num_workers=0, max_samples=128)
    SegmentationAnalysisManager.plot_analysis(dataset, num_workers=0)


if __name__ == "__main__":
    Fire(main)
