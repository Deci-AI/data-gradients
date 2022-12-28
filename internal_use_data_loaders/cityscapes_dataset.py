import os
from typing import List

from PIL import Image
from torch.utils.data import Dataset


class CityScapesDataSet(Dataset):
    CLASS_ID_TO_NAMES = {7: 'road', 8: 'sidewalk', 11: 'building', 12: 'wall', 13: 'fence', 17: 'pole',
                         19: 'traffic light', 20: 'traffic sign', 21: 'vegetation', 22: 'terrain', 23: 'sky',
                         24: 'person', 25: ' rider', 26: 'car', 27: 'truck', 28: 'bus', 31: 'train', 32: 'motorcycle',
                         33: 'bicycle'}
    NUM_CLASSES = 19
    IGNORE_LABELS = [-1, 0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]

    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.images_labels = self._read_annotations_file(image_set)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_labels)

    def __getitem__(self, index):
        current_line = self.images_labels[index].strip().split("\t")

        img_path = os.path.join(self.root, current_line[0].strip())
        label_path = os.path.join(self.root, current_line[1].strip())
        image = Image.open(img_path)
        label = Image.open(label_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def _read_annotations_file(self, image_set) -> List[str]:
        file = os.path.join(self.root, f"lists/{image_set}.lst")
        with open(file) as f:
            lines = f.readlines()
        return lines
