from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from example_dataset.bdd_dataset import BDDDataset
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager


if __name__ == "__main__":
    """
    Example script for running the Deci-Dataset-Analyzer tool.
    Arguments required for SegmentationAnalysisManager() are:
        train_data  -> An Iterable (i.e., torch data loader) containing train data
        val_data    -> An Iterable (i.e., torch data loader) containing valid data
        num_classes -> Number of valid classes
    Also if there are ignore labels, please pass them as a List[int].
    Default ignore label will be [0] as for background only.

    Example of use (CityScapes):
        train_dataset = CityScapesDataSet(root=dataset_root,
                                          image_set='train',
                                          transform=transforms_list)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        number_of_classes = 19
        ignore_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

    """
    # Create torch DataSet
    train_dataset = BDDDataset(
        data_folder="../example_dataset/bdd_example",
        split="train",
        transform=Compose([ToTensor()]),
        target_transform=Compose([ToTensor()]),
    )
    val_dataset = BDDDataset(
        data_folder="../example_dataset/bdd_example",
        split="val",
        transform=Compose([ToTensor()]),
        target_transform=Compose([ToTensor()]),
    )

    # Create torch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8)
    val_loader = DataLoader(val_dataset, batch_size=8)

    da = SegmentationAnalysisManager(
        report_title="Testing Data-Gradients",
        train_data=train_loader,
        val_data=val_loader,
        class_names=BDDDataset.CLASS_NAMES,
        class_names_to_use=BDDDataset.CLASS_NAMES[1:5],
        # Optionals
        images_extractor=None,
        labels_extractor=None,
        threshold_soft_labels=0.5,
        batches_early_stop=75,
    )

    da.run()
