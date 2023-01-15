# Data Gradients
## What is this?
The Deci`s dataset analyzer tool provide valuable information about your dataset
#### Benefits 
1. User could validate his data, find corruptions, check diversity and many more. 
2. Deci will use this meta-data as an input to our NAS, which results in a better model found, suited for the user's specific data.
#### What we DO collect
Metadata and statistics describing your data: histograms, heat maps and such.
#### What we DO NOT collect
Images themselves, Labels themselves, annotations, locations of each object. Any actual data.
You can also censor any classes you`de like, can hide class names and can remove features you are not interested in publishing. 


#### What is MUST-HAVE

* Train set data-loader that compatible to the [available input types](#available-input-types)
* Number of valid classes (in the binary case, number of classes will be 1 while 0 will be ignored)

#### What is Optional

* Validation set data-loader that compatible to the [available input types](#Available-input-types)
* Class ID-to-name mapping (in a form of a dictionary)
* Number of samples to visualize (will output only on tensorboard, can pass 0 if you prefer not to visualize)

<br>


<details>
    <summary> Available input types     </summary>



### Iterables
Python iterables objects implement the `next()` method for getting next object from iterator.
<br>
Can be ``torch.dataloader``, but not must.

### Images & Labels Objects
We support various of types for handling images or labels:
* `torch.Tensor`
* `numpy.ndarray`
* `PIL.Image`
* `Python Dictionary` (See [Python Dictionary Handling](#Python dictionary handling]))

<br>
<pre>
<details>
<summary>My dataset returns dictionary</summary>

```python
def __getitem__(...):
    return {'my_images': images: torch.Tensor,
            'my_labels': labels: numpy.ndarray,
            'my_extras': extras: List[str]
            }
```
OR
```python
def __getitem__(...):
    return images: torch.Tensor, {'my_labels': labels, 'my_other_labels': other_labels, 'labels_paths': labels_paths}
```
OR
```python
def __getitem__(...):
    return {'bgr_images': bgr_images, 'grayscale_images': grayscale_images}, labels: torch.Tensor
```
#### Python dictionary handling
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
</pre>
<br>
<pre>
<details>
<summary> My dataset returns a tuple</summary>

```python
def __getitem__(...):
    return images, labels
```
</details>
</pre>
<br>
<pre>
<details>

<summary> My dataset requires custom support </summary>

In that case, you can pass the manager a Callable (lambda or function), which handles images and labels separately.

```python
def images_extractor(x):
    x = x['images']['bgr_images']
    x /= 255.
    return x

labels_extractor = lambda x: (x['labels']['masks'] / 255.)

da = SegmentationAnalysisManager(
    train_data=train_loader,
    val_data=val_loader,
    images_extractor=images_extractor,
    labels_extarctor=labels_extractor)
```
</details>
</pre>

<br>


</details>

<br>
<details>
<summary>
Our point of view on augmentations
</summary>
<br>
Using this tool will have different benefits working with data augmentations, and without.

Augmented data will give us a better visualization of the model's point of view of the data, which will be more realistic in terms of finding problems with the training data, etc.

Raw data could be a stronger validation for the data aggregating, labeling and diversity of it.

Both options are good, but it is more important for us to see what the model will see on his training.

</details>

<br>
<details>
    <summary>How to use</summary>




### 1. Install data-gradients

```bash
pip install data_gradients-X.Y.Z-py3-none-any.whl
```
### 2. Install requirements

```bash
pip install -r requirements.txt 
```
### 3. Run analysis manager

```python
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, ToTensor, Compose

from src import SegmentationAnalysisManager
from data.bdd_dataset import BDDDataset

# Create torch DataSet
train_dataset = BDDDataset(data_folder="src/data_gradients/example_dataset/bdd_example", split='train', transform=Compose([ToTensor()]))
val_dataset = BDDDataset(data_folder="src/data_gradients/example_dataset/bdd_example", split='val', transform=Compose([ToTensor()]))

# Create torch DataLoader
train_loader = DataLoader(train_dataset, batch_size=8)
val_loader = DataLoader(val_dataset, batch_size=8)

da = SegmentationAnalysisManager(train_data=train_loader,
                                 val_data=val_loader,
                                 num_classes=BDDDataset.NUM_CLASSES)

da.run()


```
### 4. After progress is finished, view results through tensorboard

```bash
tensorboard --logdir=logs --bind_all
```
Click on link and view results:

``TensorBoard 2.11.0 at http://localhost:6007/ (Press CTRL+C to quit)``

</details>
<br>

<details>
<summary>
Output Example
</summary>

![Example output 1](data/example_outputs/output_example1.png)

![Example output 2](data/example_outputs/output_example2.png)

![Example output 3](data/example_outputs/output_example3.png)

</details>
