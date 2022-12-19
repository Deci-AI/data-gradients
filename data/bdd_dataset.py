import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

NORMALIZATION_MEANS = [.485, .456, .406]
NORMALIZATION_STDS = [.229, .224, .225]


class BDDDataset(Dataset):
    """
    PyTorch Dataset implementation of the BDD100K dataset.
    The BDD100K data and annotations can be obtained at https://bdd-data.berkeley.edu/.
    """
    NUM_CLASSES = 20
    IGNORE_LABELS = [0, 19]

    def __init__(self, data_folder, split: str, ignore_label=19, transform=transforms.Compose([])):
        """
        :param data_folder: Folder where data files are stored
        :param split: 'train' or 'test'
        :param ignore_label: label to ignore for certain metrics
        """
        data_location = os.path.join(data_folder, split)
        files_list = os.listdir(data_location)
        self.ignore_label = ignore_label
        self.samples_fn = []
        for f in files_list:
            if f[-3:] == "jpg":

                self.samples_fn.append([os.path.join(data_location, f), os.path.join(data_location, f[0:-3] + "png")])

        self.transforms = transform

    @staticmethod
    def sample_transform(image):
        """
        sample_transform - Transforms the sample image
          :param image: The input image to transform
          :return:    The transformed image
        """
        sample_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEANS, NORMALIZATION_STDS)])
        return sample_transform(image)

    @staticmethod
    def target_transform(target):
        """
        target_transform - Transforms the sample image
          :param target: The target mask to transform
          :return:    The transformed target mask
        """
        return torch.from_numpy(np.array(target)).long()

    def __getitem__(self, i):
        image = Image.open(self.samples_fn[i][0]).convert('RGB')
        label = Image.open(self.samples_fn[i][1])
        # if self.transforms:
        #     t = self.transforms({"image": image, "mask": label})
        #     image, label = t['image'], t['mask']

        image_tensor, label = self.sample_transform(image), self.target_transform(label)
        label = np.array(label)
        label[label == 255] = self.ignore_label

        label_tensor = torch.from_numpy(label).long()
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.samples_fn)