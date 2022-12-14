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
train_dataloader, val_dataloader = DataLoaders().get_dataloader(dataset="sbd")
train_data_iterator, val_data_iterator = iter(train_dataloader), iter(val_dataloader)
```
### 4. At `main.py` import dataset and run script

```python
import hydra
from omegaconf import DictConfig

from manager import AnalysisManager
from data_loaders.get_torch_loaders import train_data_iterator, val_data_iterator


@main.main(version_base=None, config_path=".", config_name='feature_extractors')
def main(cfg: DictConfig) -> None:
    da = AnalysisManager(cfg, train_data_iterator, val_data_iterator)
    da.run()


if __name__ == '__main__':
    main()
```
### 5. After progress is finished, view results through tensorboard

```bash
tensorboard --logdir=logs/train_data --bind_all
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