"""This is an example will require you to install the super-gradients package.

It shows how DataGradients can be used on top of Datasets provided by SuperGradients.

Required step:
```
pip install super-gradients
```
"""

import numpy as np
from torch.utils.data import DataLoader

# Note: This example will require you to install the super-gradients package
from super_gradients.training.datasets import YoloDarknetFormatDetectionDataset

from data_gradients.managers.detection_manager import DetectionAnalysisManager


class PadTarget:
    """Transform targets (Compatible with Sg DetectionDatasets)"""

    def __init__(self, max_targets: int):
        self.max_targets = max_targets

    def __call__(self, sample):
        targets = sample["target"]
        targets = np.ascontiguousarray(targets, dtype=np.float32)
        padded_targets = np.zeros((self.max_targets, targets.shape[-1]))
        padded_targets[range(len(targets))[: self.max_targets]] = targets[: self.max_targets]
        padded_targets = np.ascontiguousarray(padded_targets, dtype=np.float32)
        sample["target"] = padded_targets
        return sample


if __name__ == "__main__":
    # Chose a dataset from Roboflow and change the data_dir and classes accordingly
    data_dir = "<path-to-avatar_recognition>"
    classes = ["Character"]

    # Create torch DataSet
    train_dataset = YoloDarknetFormatDetectionDataset(
        data_dir=data_dir,
        images_dir="train/images",
        labels_dir="train/labels",
        classes=classes,
        transforms=[PadTarget(max_targets=50)],
    )
    val_dataset = YoloDarknetFormatDetectionDataset(
        data_dir=data_dir,
        images_dir="valid/images",
        labels_dir="valid/labels",
        classes=classes,
        transforms=[PadTarget(max_targets=50)],
    )

    # Create torch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8)
    val_loader = DataLoader(val_dataset, batch_size=8)

    analyzer = DetectionAnalysisManager(
        train_data=train_loader,
        val_data=val_loader,
        n_classes=len(classes),
    )

    analyzer.run()
