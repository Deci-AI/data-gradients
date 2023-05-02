import numpy as np
from super_gradients.training.datasets import YoloDarknetFormatDetectionDataset

from torch.utils.data import DataLoader

from data_gradients.managers.detection_manager import DetectionAnalysisManager


class PadTarget:
    def __init__(self, max_targets):
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
    data_dir = "/Users/Louis.Dupont/Downloads/avatar_recognition.v2-release.yolov8"
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
        samples_to_visualize=0,
    )

    analyzer.run()
