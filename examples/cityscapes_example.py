from fire import Fire
from torchvision.datasets.cityscapes import Cityscapes
from data_gradients.dataset_adapters import TorchvisionCityscapesSegmentationAdapter
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager


def main(root="e:/cityscapes"):
    train_dataset = Cityscapes(
        root=root,
        split="train",
        target_type="semantic",
    )
    val_dataset = Cityscapes(
        root=root,
        split="val",
        target_type="semantic",
    )

    dataset = {
        "train": TorchvisionCityscapesSegmentationAdapter(train_dataset),
        "val": TorchvisionCityscapesSegmentationAdapter(val_dataset),
    }

    SegmentationAnalysisManager.plot_analysis(dataset, num_workers=24)


if __name__ == "__main__":
    Fire(main)
