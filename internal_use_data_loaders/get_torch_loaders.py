from typing import Tuple, Optional

import torchvision.transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, InterpolationMode
from torch.utils.data import DataLoader

from data.bdd_dataset import BDDDataset
from internal_use_data_loaders.cityscapes_dataset import CityScapesDataSet
from internal_use_data_loaders.pp_humanseg_14k_dataset import PPHumanSegDataSet


class DataLoaders:
    def __init__(self, batch_size: int = 16):
        self._transforms = torchvision.transforms.Compose([ToTensor()]) #, Resize(256, interpolation=InterpolationMode.NEAREST)])
        self._target_transforms = torchvision.transforms.Compose([ToTensor()]) #, Resize(384, interpolation=InterpolationMode.NEAREST)])
        self._batch_size = batch_size

    def _dataset_to_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=True, num_workers=0)

    def bdd(self):
        dataset_root = "/Users/tomerkeren/workspace/deci-dataset-analyzer/data/bdd100k"
        train = BDDDataset(data_folder=dataset_root, split='train', transform=self._transforms)
        val = BDDDataset(data_folder=dataset_root, split='val', transform=self._transforms)

        train_loader = self._dataset_to_dataloader(train)
        val_loader = self._dataset_to_dataloader(val)
        return train_loader, val_loader

    def sbd(self):
        train = datasets.SBDataset(
            root="data/sbd",
            image_set="train",
            mode="segmentation",
            download=False,
        )

        val = datasets.SBDataset(
            root="data/sbd",
            image_set="val",
            mode="segmentation",
            download=False,
        )

        train_dataloader = self._dataset_to_dataloader(train)
        val_dataloader = self._dataset_to_dataloader(val)
        return train_dataloader, val_dataloader

    def pp_human(self):
        dataset_root = "/Users/tomerkeren/workspace/deci-dataset-analyzer/data/pp_humanseg/PP-HumanSeg14K"
        train = PPHumanSegDataSet(root=dataset_root,
                                  image_set='train',
                                  transform=self._transforms,
                                  target_transform=self._target_transforms)

        val = PPHumanSegDataSet(root=dataset_root,
                                image_set='val',
                                transform=torchvision.transforms.Compose([ToTensor()]),
                                target_transform=torchvision.transforms.Compose([ToTensor()]))

        train_dataloader = self._dataset_to_dataloader(train)
        val_dataloader = self._dataset_to_dataloader(val)
        return train_dataloader, val_dataloader

    def cityscapes(self):
        dataset_root = "/Users/tomerkeren/workspace/deci-dataset-analyzer/data/cityscapes"
        train = CityScapesDataSet(root=dataset_root,
                                  image_set='train',
                                  transform=self._transforms,
                                  target_transform=self._transforms)
        val = CityScapesDataSet(root=dataset_root,
                                image_set='val',
                                transform=self._target_transforms,
                                target_transform=self._target_transforms)

        train_dataloader = self._dataset_to_dataloader(train)
        val_dataloader = self._dataset_to_dataloader(val)

        return train_dataloader, val_dataloader

    def get_dataloader(self, dataset: str) -> Tuple[DataLoader, Optional[DataLoader]]:
        if dataset == "sbd":
            return self.sbd()

        elif dataset == 'bdd':
            return self.bdd()

        elif dataset == "pp_human":
            return self.pp_human()

        elif dataset == 'cityscapes':
            return self.cityscapes()
        else:
            raise NotImplementedError



dataloader = DataLoaders(batch_size=16)


## CityScapes
train_loader, val_loader = dataloader.get_dataloader(dataset="cityscapes")
num_classes = CityScapesDataSet.NUM_CLASSES
ignore_labels = CityScapesDataSet.IGNORE_LABELS
class_id_to_name = CityScapesDataSet.CLASS_ID_TO_NAMES

## PPHuman
# train_loader, val_loader = dataloader.get_dataloader(dataset="pp_human")
# num_classes = PPHumanSegDataSet.NUM_CLASSES
# ignore_labels = getattr(PPHumanSegDataSet, 'IGNORE_LABELS', None)
# class_id_to_name = PPHumanSegDataSet.CLASS_ID_TO_NAMES


## BDD
# train_loader, val_loader = dataloader.get_dataloader(dataset="bdd")
# num_classes = BDDDataset.NUM_CLASSES
# ignore_labels = BDDDataset.IGNORE_LABELS
# class_id_to_name = BDDDataset.CLASS_ID_TO_NAMES


## SBD
# SBDDataset = datasets.SBDataset(root="data/sbd",
#                                 image_set="train",
#                                 mode="segmentation",
#                                 download=False)
# CLASS_ID_TO_NAMES = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
#                      7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
#                      14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train',
#                      20: 'tvmonitor'}
# train_loader, val_loader = dataloader.get_dataloader(dataset="sbd")
# num_classes = SBDDataset.num_classes
# ignore_labels = None
# class_id_to_name = CLASS_ID_TO_NAMES
