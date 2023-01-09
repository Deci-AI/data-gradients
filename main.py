from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from data_gradients.example_dataset.bdd_dataset import BDDDataset
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager


if __name__ == "__main__":
    """
    Main script for running the Deci-Dataset-Analyzer tool.
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
    train_dataset = BDDDataset(data_folder="src/data_gradients/example_dataset/bdd_example", split='train',
                               transform=Compose([ToTensor()]), target_transform=Compose([ToTensor()]))
    val_dataset = BDDDataset(data_folder="src/data_gradients/example_dataset/bdd_example", split='val',
                             transform=Compose([ToTensor()]), target_transform=Compose([ToTensor()]))

    # Create torch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    num_classes = BDDDataset.NUM_CLASSES
    ignore_labels = BDDDataset.IGNORE_LABELS
    class_id_to_name = BDDDataset.CLASS_ID_TO_NAMES

    # from internal_use_data_loaders.get_torch_loaders import *

    da = SegmentationAnalysisManager(train_data=train_loader,
                                     val_data=val_loader,
                                     num_classes=num_classes,
                                     ignore_labels=ignore_labels,
                                     # Optionals
                                     id_to_name=class_id_to_name,
                                     samples_to_visualize=2,
                                     images_extractor=None,
                                     labels_extractor=None,
                                     # batches_early_stop is a temporary optional argument for debugging purpose
                                     batches_early_stop=999)

    da.run()
