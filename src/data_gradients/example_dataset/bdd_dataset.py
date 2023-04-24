import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

NORMALIZATION_MEANS = [0.485, 0.456, 0.406]
NORMALIZATION_STDS = [0.229, 0.224, 0.225]


class BDDDataset(Dataset):
    """
    PyTorch Dataset implementation of the BDD100K dataset.
    The BDD100K data and annotations can be obtained at https://bdd-data.berkeley.edu/.
    """

    n_classes = 19
    IGNORE_LABELS = [19]
    CLASS_ID_TO_NAMES = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
    }

    def __init__(
        self,
        data_folder,
        split: str,
        ignore_label=19,
        transform=transforms.Compose([]).transforms,
        target_transform=transforms.Compose([]),
    ):
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

                self.samples_fn.append(
                    [
                        os.path.join(data_location, f),
                        os.path.join(data_location, f[0:-3] + "png"),
                    ]
                )

        self.transforms = transform
        self.target_transforms = transform

    def get_target(self, target):
        # Mask as normalized tensor
        mask = self.transforms(target)
        mask[mask == 1.0] = self.ignore_label / 255.0
        return mask

    def __getitem__(self, i):
        image = Image.open(self.samples_fn[i][0]).convert("RGB")
        label = Image.open(self.samples_fn[i][1])
        if self.transforms:
            image = self.transforms(image)
            label = self.get_target(label)

        return image, label

    def __len__(self):
        return len(self.samples_fn)
