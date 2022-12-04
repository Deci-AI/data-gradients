import torchvision.transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, CenterCrop
from torch.utils.data import DataLoader


dataset_path = "/Users/tomerkeren/data/PP-HumanSeg14K/"


training_data = datasets.SBDataset(
    root="data",
    image_set="train",
    mode="segmentation",
    download=False,
    transforms=torchvision.transforms.Compose([ToTensor(), CenterCrop(512)])
)

val_data = datasets.SBDataset(
    root="data",
    image_set="val",
    mode="segmentation",
    download=False,
    transforms=torchvision.transforms.Compose([ToTensor(), CenterCrop(512)])
)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)

# Train
# Aeroplane
# Chair
# Bottle
# Dining Table
# Potted Plant
# TV/Monitor
# Sofa
# Bird
# Cat
# Cow
# Dog
# Horse
# Sheep
label_to_class = {0: 'aeroplane',
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
