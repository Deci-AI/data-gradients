# <Your Library Name>

Data-Gradients is an open-source Exploratory Data Analysis (EDA) library specifically designed for computer vision applications. 
It automatically extracts features from your datasets and combines them all into a single user-friendly report. 

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

Here, list out the features of your library. If there are many, consider using bullet points for each one. For example:

- Feature 1: Description of feature 1
- Feature 2: Description of feature 2
- etc.

## Installation

Provide clear instructions on how to install your library. For example:

To install Data-Gradients, you first need to clone the repository

```
git clone https://github.com/Deci-AI/data-gradients
```

Move the local directory
```
cd data-gradients
```

Install Data-Gradients
```
pip install .
```

## Quick Start

First, prepare your train_data and val_data.
This can be a pytorch dataset, dataloader or any type of data iterable.

The output of these data iterables can be anything. 
Data-Gradients will try to find out how you stored your image and labels.
If something cannot be automatically determined, you will be asked to provide some extra information through a text input.
In some extreme cases, the process will crash and ask you to provide a clear adapter.

### Image Adapter
You can provide an Image Adapter function: `images_extractor(data: Any) -> torch.Tensor:`

- `data` being the output of the dataset/dataloader that you provided.
- The function should return a Tensor representing your image(s):
  - `(BS, C, H, W)`, `(BS, H, W, C)`, `(BS, H, W)` for batch, with `C`: number of channels (3 for RGB)
  - `(C, H, W)`, `(H, W, C)`, `(H, W)` for single image, with `C`: number of channels (3 for RGB)


---
class LabelsExtractorError(Exception):
    def __init__(self):
        msg = (
            "\n\nERROR: Something went wrong when extracting Labels!\n"
            "Please implement and pass to the config the following function `labels_extractor(data: Any) -> torch.Tensor:`\n"
            "Make sure to:\n"
            "     - `data` being the output of the dataset/dataloader that you provided.\n"
            "     - The function should return a Tensor representing your label(s). Possible formats are:\n"
            "             - (BS, C, H, W), (BS, H, W, C), (BS, H, W) for batch in segmentation.\n"
            "             - (C, H, W), (H, W, C), (H, W) for single image in segmentation.\n"
        )
        super().__init__(msg)

#### Object Detection
If your train_data/val_data returns targets/labels 
```python
from data_gradients.managers.detection_manager import DetectionAnalysisManager

train_loader = ...
val_loader = ...
class_names = ...

analyzer = DetectionAnalysisManager(
    report_title="Testing Data-Gradients",
    train_data=train_loader,
    val_data=val_loader,
    class_names=class_names,
)

analyzer.run()
```

#### Segmentation
```python
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager 

train_loader = ...
val_loader = ...
class_names = ...

analyzer = SegmentationAnalysisManager(
    report_title="Testing Data-Gradients",
    train_data=train_loader,
    val_data=val_loader,
    class_names=class_names,
)

analyzer.run()
```

## Examples

Provide or link to more extensive examples if they're too long to fit within the ReadMe. For example:

Check out these more detailed examples to get a sense of what you can do with <Your Library Name>:

- [Example 1](link to example 1)
- [Example 2](link to example 2)

## Contributing

Encourage others to contribute to your project. Describe how they can do that. For example:

Contributions to <Your Library Name> are very welcome! If you have a feature request, bug report, or have written code you want to contribute, please first read our [Contributing Guide](link to contributing guide) and then [open an issue](link to new issue page on your GitHub).

## License

Specify the license under which you're releasing your code. For example:

<Your Library Name> is licensed under the [MIT License](link to the license).

## Acknowledgements

Here, acknowledge the people who helped you and/or inspired you to create this library.

- [Name 1](link to their GitHub or website)
- [Name 2](link to their GitHub or website)
- etc.

```

Remember to replace the placeholders `<Your Library Name>`, `function1`, `function2`, etc., with actual content. Additionally, the template assumes you're distributing your library via pip and that your project is hosted on GitHub, so adjust those details as needed.


# Data Gradients

Pyhon based open source project
- EDA tool
- Extract features from Computer Vision datasets
  - Segmentation
  - Detection


- Designed to help Practitioners to better understand their data
    - See if anything is wrong in the dataset
    - See if the train/validation splits make sense (For instance if a cass is not represented in the training we won't be able to learn it)
    - Evaluate dataset quality - If you are looking for a dataset to train on
    - Find proxy datasets - that includes similar features to the original dataset


# How to use
- Designed to run in a few clicks
  - Works with datasets, dataloaders and any data iterator 
  

- Generate a PDF report with
  - Graph representing the feature (distribution, violinplot, heatmap, etc)
  - Short description of what can be seen in that feature











## What is this?
A Python-based repository that extracts meta data from your data loaders and visualizes it.
#### The benefits of Data Gradients
With Data Gradients, you can analyze your data in order to gain valuable insights.
1. Data validation: detecting corruption, ensuring diversity, and more.
2. Metadata extraction for maximizing the customized results for your architecture search

#### What does the data analyzer tool extract?
Statistics and metadata describe your data: histograms, heat maps, etc.

#### What doesn’t the data analyzer tool extract?
Images, labels, annotations, and locations of each object.
In addition, you can censor any classes you want, hide class names, and remove features you don’t want the tool to extract.

#### What does the tool output?
The tool extracts statistics and metadata into a TB file and a corresponding TXT file. The metadata and statistics in both files are the same.

#### What is a MUST-HAVE in order to use the tool?

* Train set data-loader that compatible to the available input types: Fill in the types
* Number of valid classes (in the binary case, the number of classes will be 1 while 0 will be ignored)

#### What is Optional in order to use the tool?

* Validation set data-loader that compatible to the available input types
* Class ID-to-name mapping (in a form of a dictionary)
* Number of samples to visualize (will output only on Tensorboard, can pass 0 if you prefer not to visualize)

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
There are different benefits to using this tool with or without data augmentations.
Using augmented data will allow us to see the model’s point of view of the data, which will be more realistic when finding problems with the training data.
Raw data, on the other hand, could provide stronger validation for data aggregation, labeling, and diversity.
There are advantages to both options, but as this tool is designed to optimize and customize the architecture for your data, we need to see what the model will see during training.

</details>

<br>
<details>
    <summary>How to use</summary>



### 1. Install data-gradients

```bash
pip install data_gradients-X.Y.Z-py3-none-any.whl
```
### 2. Run analysis manager

```python
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager

# Create torch DataLoader
train_loader = YourDataLoader(train_dataset, batch_size=batch_size)
val_loader = YourDataLoader(val_dataset, batch_size=batch_size)

da = SegmentationAnalysisManager(train_data=train_loader,
                                 val_data=val_loader,
                                 num_classes=num_classes)

da.run()


```
### 3. After progress is finished, view results through tensorboard

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
