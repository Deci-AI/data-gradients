from typing import Tuple, Optional

import torchvision.transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, CenterCrop
from torch.utils.data import DataLoader

from data.bdd_dataset import BDDDataset
from internal_use_data_loaders.cityscapes_dataset import CityScapesDataSet
from internal_use_data_loaders.pp_humanseg_14k_dataset import PPHumanSegDataSet

# TODO: Clean up all methods but "bdd"
# Make dataset root path relative to project's, pointing at small example bdd dataset


class DataLoaders:
    def __init__(self):
        self._transforms = torchvision.transforms.Compose([ToTensor(), CenterCrop(512)])
        self._batch_size = 32

    def _dataset_to_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

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
            transforms=self._transforms
        )

        val = datasets.SBDataset(
            root="data/sbd",
            image_set="val",
            mode="segmentation",
            download=False,
            transforms=self._transforms
        )

        train_dataloader = self._dataset_to_dataloader(train)
        val_dataloader = self._dataset_to_dataloader(val)
        return train_dataloader, val_dataloader

    def pp_human(self):
        dataset_root = "/Users/tomerkeren/workspace/deci-dataset-analyzer/data/pp_humanseg/PP-HumanSeg14K"
        train = PPHumanSegDataSet(root=dataset_root,
                                  image_set='train',
                                  transform=self._transforms)

        val = PPHumanSegDataSet(root=dataset_root,
                                image_set='val',
                                transform=self._transforms)
        train_dataloader = self._dataset_to_dataloader(train)
        val_dataloader = self._dataset_to_dataloader(val)
        return train_dataloader, val_dataloader

    def cityscapes(self):
        dataset_root = "/Users/tomerkeren/workspace/deci-dataset-analyzer/data/cityscapes"
        train = CityScapesDataSet(root=dataset_root,
                                  image_set='train',
                                  transform=self._transforms)
        val = CityScapesDataSet(root=dataset_root,
                                image_set='val',
                                transform=self._transforms)
        train_dataloader = self._dataset_to_dataloader(train)
        val_dataloader = self._dataset_to_dataloader(val)
        return train_dataloader, val_dataloader

    def get_dataloader(self, dataset: str) -> Tuple[DataLoader, Optional[DataLoader]]:
        if dataset == "sbd":
            return self.sbd()

        elif dataset == 'bdd':
            return self.bdd()
        elif dataset == "kitti":
            raise NotImplementedError

        elif dataset == "voc_seg":
            raise NotImplementedError

        elif dataset == "pp_human":
            return self.pp_human()

        elif dataset == 'cityscapes':
            return self.cityscapes()
        else:
            raise NotImplementedError


sbd = True
kitti = False
voc_segmentation = False
# [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
# sbd, pp_human, cityscapes, bdd
# 20,  1,        19,         40
train_dataloader, val_dataloader = DataLoaders().get_dataloader(dataset="bdd")
train_data_iterator, val_data_iterator = iter(train_dataloader), iter(val_dataloader)
num_classes = 20
ignore_labels = [0]
