import os
from typing import List

from PIL import Image
from torch.utils.data import Dataset


class PPHumanSegDataSet(Dataset):
    CLASS_LABELS = {0: "background", 1: "person"}

    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.images_labels = self._read_annotations_file(image_set)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_labels)

    def __getitem__(self, index):
        current_line = self.images_labels[index].split(" ")
        img_path = os.path.join(self.root, current_line[0].strip())
        label_path = os.path.join(self.root, current_line[1].strip())
        image = Image.open(img_path)
        label = Image.open(label_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.transform is not None:
            label = self.transform(label)
        return image, label

    def _read_annotations_file(self, image_set) -> List[str]:
        file = os.path.join(self.root, f"{image_set}.txt")
        with open(file) as f:
            lines = f.readlines()
        return lines
