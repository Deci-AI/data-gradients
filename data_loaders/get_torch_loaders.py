from typing import Tuple, Optional

import torchvision.transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, CenterCrop
from torch.utils.data import DataLoader

from data_loaders.pp_humanseg_14k_dataset import PPHumanSegDataSet


class DataLoaders:
    def __init__(self):
        self._transforms = torchvision.transforms.Compose([ToTensor(), CenterCrop(512)])
        self._batch_size = 64

    def _dataset_to_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

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

    def get_dataloader(self, dataset: str) -> Tuple[DataLoader, Optional[DataLoader]]:
        if dataset == "sbd":
            return self.sbd()

        elif dataset == "kitti":
            raise NotImplementedError

        elif dataset == "voc_seg":
            raise NotImplementedError

        elif dataset == "pp_human":
            return self.pp_human()
        else:
            raise NotImplementedError


sbd = True
kitti = False
voc_segmentation = False


# elif kitti:
#     def get_only_seg(target):
#         print(target.keys())
#         pass
#
#
#     training_data = Kitti(root='data/Kitty',
#                           download=True,
#                           train=True,
#                           transform=torchvision.transforms.Compose([ToTensor(), CenterCrop(512)]))
#     val_data = Kitti(root='data/Kitty',
#                      download=True,
#                      train=False,
#                      transform=torchvision.transforms.Compose([ToTensor(), CenterCrop(512)]),
#                      target_transform=get_only_seg)

# elif voc_segmentation:
#     training_data = VOCSegmentation(root="data/VOC",
#                                     image_set='train',
#                                     download=True,
#                                     transform=torchvision.transforms.Compose([ToTensor(), CenterCrop(512)]))
#     val_data = VOCSegmentation(root="data/VOC",
#                                image_set='val',
#                                download=True,
#                                transform=torchvision.transforms.Compose([ToTensor(), CenterCrop(512)]))
# else:
#     print("No dataset has been chosen")
#     exit(0)
#



sbd_label_to_class = {0: 'aeroplane',
                      1: 'bicycle',
                      2: 'bird',
                      3: 'boat',
                      4: 'bottle',
                      5: 'bus',
                      6: 'car',
                      7: 'cat',
                      8: 'chair',
                      9: 'cow',
                      10: 'diningtable',
                      11: 'dog',
                      12: 'horse',
                      13: 'motorbike',
                      14: 'person',
                      15: 'pottedplant',
                      16: 'sheep',
                      17: 'sofa',
                      18: 'train',
                      19: 'tvmonitor'
                      }
