from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, ToTensor, Compose

from data.bdd_dataset import BDDDataset


# Create torch DataSet
train_dataset = BDDDataset(data_folder="../../data/bdd_example", split='train',
                           transform=Compose([ToTensor(), CenterCrop(512)]))

# Create torch DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

num_classes = BDDDataset.NUM_CLASSES
ignore_labels = BDDDataset.IGNORE_LABELS
