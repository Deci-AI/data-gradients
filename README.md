# deci-dataset-analyzer
<details>
    <summary>How to use</summary>

### 1. clone GitHub repository
```bash
git clone https://github.com/Deci-AI/deci-dataset-analyzer
```
### 2. install requirements
```bash
pip install -r requirements.txt
```
### 3. Connect dataset with Python-Iterables objects

```python
from torchvision import datasets
from torch.utils.data import DataLoader

train_dataset = datasets.SBDataset(root="data/sbd",
                                   image_set="train",
                                   mode="segmentation")
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

```
### 4. At `main.py` import dataset and run script

```python
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, ToTensor, Compose

from src import SegmentationAnalysisManager
from data.bdd_dataset import BDDDataset

# Create torch DataSet
train_dataset = BDDDataset(data_folder="data/bdd_example", split='train', transform=Compose([ToTensor(), CenterCrop(512)]))
val_dataset = BDDDataset(data_folder="data/bdd_example", split='val', transform=Compose([ToTensor(), CenterCrop(512)]))

# Create torch DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

da = SegmentationAnalysisManager(train_data=train_loader,
                                 val_data=val_loader,
                                 num_classes=BDDDataset.NUM_CLASSES,
                                 ignore_labels=BDDDataset.IGNORE_LABELS)
da.run()

```
### 5. After progress is finished, view results through tensorboard

```bash
tensorboard --logdir=logs --bind_all
```
Click on link and view results:

``TensorBoard 2.11.0 at http://localhost:6007/ (Press CTRL+C to quit)``

</details>

<details>
    <summary>Available input types</summary>

### Iterables
Python iterables objects implement the `next()` method for getting next object from iterator.
For now, we only support the situation where the objects are:
* Tuple
* Two objects

where the two objects should be in this form:
``
(images, labels)
``
### Tuples Objects
We support various of object types:
* `torch.Tensor`
* `numpy.ndarray`
* `PIL.Image`
* `Python Dictionary`

As for the python dictionary, because of the various ways of getting
an item out of it, we will activate an interactive small utility
for extracting the right object out of the dictionary. This tool will map all the 
objects that this dictionary holds, and will ask you to choose which one is
the right one, either for "images" or for "labels".

Example:
```yaml
{
     all_labels: {
          not_good_torch_labels: Tensor ⓪,
          not_good_np_labels: ndarray ①,
          good_torch_labels: Tensor ②
     },
     something_other_then_labels: ndarray ③
}

prompt >> which one of the yellow items is your required data?
user input >> 2
```

</details>