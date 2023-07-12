# DataGradients
<div align="center">
<p align="center">
  <a href="https://github.com/Deci-AI/super-gradients#prerequisites"><img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue" /></a>
  <a href="https://pypi.org/project/data-gradients/"><img src="https://img.shields.io/pypi/v/data-gradients" /></a>
  <a href="https://github.com/Deci-AI/data-gradients/releases"><img src="https://img.shields.io/github/v/release/Deci-AI/data-gradients" /></a>
  <a href="https://github.com/Deci-AI/data-gradients/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>
</p>   
</div>

DataGradients is an open-source python based library specifically designed for computer vision dataset analysis. 

It automatically extracts features from your datasets and combines them all into a single user-friendly report. 

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
   - [Prerequisites](#prerequisites)
   - [Dataset Analysis](#dataset-analysis)
   - [Report](#report)
- [Feature Configuration](#feature-configuration)
- [Dataset Adapters](#dataset-adapters)
   - [Image Adapter](#image-adapter)
   - [Label Adapter](#label-adapter)
   - [Example](#example)
- [License](#license)

## Features
- Image-Level Evaluation: DataGradients evaluates key image features such as resolution, color distribution, and average brightness.
- Class Distribution: The library extracts stats allowing you to know which classes are the most used, how many objects do you have per image, how many image without any label, ...
- Heatmap Generation: DataGradients produces heatmaps of bounding boxes or masks, allowing you to understand if the objects are positioned in the right area.
- And [many more](./documentation/feature_description.md)!

<div align="center">
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_image_stats.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_image_stats.png" width="250px"></a>
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_mask_sample.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_mask_sample.png" width="250px"></a>
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_classes_distribution.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_classes_distribution.png" width="250px"></a>
  <p><em>Example of pages from the Report</em>
</div>

<div align="center">
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationBoundingBoxArea.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationBoundingBoxArea.png" width="375px"></a>
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationBoundingBoxResolution.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationBoundingBoxResolution.png" width="375px"></a>
  <br />
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationClassFrequency.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationClassFrequency.png" width="375px"></a>
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationComponentsPerImageCount.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationComponentsPerImageCount.png" width="375px"></a>
  <p><em>Example of specific features</em>
</div>

## Examples 
[COCO 2017 Detection report](documentation/assets/Report_COCO.pdf)

[Cityscapes Segmentation report](documentation/assets/Report_CityScapes.pdf)

<table style="border: 0">
  <tr>
    <td><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/colab.png" width="80pt"></td>
    <td><a href="https://colab.research.google.com/drive/1dswgeK0KF-n61p6ixRdFgbQKHEtOu8SE?usp=sharing"> Example notebook on Colab</a></td>
  </tr>
</table>


## Installation
You can install DataGradients directly from the GitHub repository.

```
pip install data-gradients
```


## Quick Start

### Prerequisites

- **Dataset**: Includes a **Train** set and a **Validation** or a **Test** set.
- **Class Names**: A list of the unique categories present in your dataset.
- **Iterable**: A method to iterate over your Dataset providing images and labels. Can be any of the following:
  - PyTorch Dataloader
  - PyTorch Dataset
  - Generator that yields image/label pairs
  - Any other iterable you use for model training/validation

Please ensure all the points above are checked before you proceed with **DataGradients**.

**Good to Know**: DataGradients will try to find out how the dataset returns images and labels.
- If something cannot be automatically determined, you will be asked to provide some extra information through a text input.
- In some extreme cases, the process will crash and invite you to implement a custom dataset adapter (see relevant section)

**Heads up**: We currently don't provide out-of-the-box dataset/dataloader implementation. 
You can find multiple dataset implementations in [PyTorch](https://pytorch.org/vision/stable/datasets.html) 
or [SuperGradients](https://docs.deci.ai/super-gradients/src/super_gradients/training/datasets/Dataset_Setup_Instructions.html). 

**Example**
``` python
from torchvision.datasets import CocoDetection

train_data = CocoDetection(...)
val_data = CocoDetection(...)
class_names = ["person", "bicycle", "car", "motorcycle", ...]
```


### Dataset Analysis
You are now ready to go, chose the relevant analyzer for your task and run it over your datasets!

**Object Detection**
```python
from data_gradients.managers.detection_manager import DetectionAnalysisManager

train_data = ...
val_data = ...
class_names = ...

analyzer = DetectionAnalysisManager(
    report_title="Testing Data-Gradients Object Detection",
    train_data=train_data,
    val_data=val_data,
    class_names=class_names,
)

analyzer.run()
```


**Semantic Segmentation**
```python
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager 

train_data = ...
val_data = ...
class_names = ...

analyzer = SegmentationAnalysisManager(
    report_title="Testing Data-Gradients Segmentation",
    train_data=train_data,
    val_data=val_data,
    class_names=class_names,
)

analyzer.run()
```

**Example**

You can test the segmentation analysis tool in the following [example](https://github.com/Deci-AI/data-gradients/blob/master/examples/segmentation_example.py)
which does not require you to download any additional data.


### Report
Once the analysis is done, the path to your pdf report will be printed.


## Feature Configuration
 
The feature configuration allows you to run the analysis on a subset of features or adjust the parameters of existing features. 
If you are interested in customizing this configuration, you can check out the [documentation](documentation/feature_configuration.md) on that topic.


## Dataset Adapters
Before implementing a Dataset Adapter try running without it, in many cases DataGradient will support your dataset without any code.

Two type of Dataset Adapters are available: `images_extractor` and `labels_extractor`. These functions should be passed to the main Analyzer function init.

```python
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager

train_data = ...
val_data = ...

# Let Assume that in this case, the  train_data and val_data return data in this format:
# (image, {"masks", "bboxes"})
images_extractor = lambda data: data[0]             # Extract the image
labels_extractor = lambda data: data[1]['masks']    # Extract the masks

# In case of segmentation. 
SegmentationAnalysisManager(
    report_title="Test with Adapters",
    train_data=train_data,
    val_data=val_data,
    images_extractor=images_extractor, 
    labels_extractor=labels_extractor, 
)

# For Detection, just change the Manager and the label_extractor definition.
```

### Image Adapter
Image Adapter functions should respect the following:

`images_extractor(data: Any) -> torch.Tensor`

- `data` being the output of the dataset/dataloader that you provided.
- The function should return a Tensor representing your image(s). One of:
  - `(BS, C, H, W)`, `(BS, H, W, C)`, `(BS, H, W)` for batch
  - `(C, H, W)`, `(H, W, C)`, `(H, W)` for single image
    - With `C`: number of channels (3 for RGB)


### Label Adapter
Label Adapter functions should respect the following: 

`labels_extractor(data: Any) -> torch.Tensor`

- `data` being the output of the dataset/dataloader that you provided.
- The function should return a Tensor representing your labels(s):
  - For **Segmentation**, one of: 
    - `(BS, C, H, W)`, `(BS, H, W, C)`, `(BS, H, W)` for batch
    - `(C, H, W)`, `(H, W, C)`, `(H, W)` for single image
      - `BS`: Batch Size
      - `C`: number of channels - 3 for RGB
      - `H`, `W`: Height and Width
  - For **Detection**, one of:
    - `(BS, N, 5)`, `(N, 6)` for batch
    - `(N, 5)` for single image
      - `BS`: Batch Size
      - `N`: Padding size
      - The last dimension should include your `class_id` and `bbox` - `class_id, x, y, x, y` for instance


### Example

Let's imagine that your dataset returns a couple of `(image, annotation)` with `annotation` as below:
``` python
annotation = [
    {"bbox_coordinates": [1.08, 187.69, 611.59, 285.84], "class_id": 51},
    {"bbox_coordinates": [5.02, 321.39, 234.33, 365.42], "class_id": 52},
    ...
]
```

Because this dataset includes a very custom type of `annotation`, you will need to implement your own custom `labels_extractor` as below:
``` python
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager

def labels_extractor(data: Tuple[PIL.Image.Image, List[Dict]]) -> torch.Tensor:
    _image, annotations = data[:2]
    labels = []
    for annotation in annotations:
        class_id = annotation["class_id"]
        bbox = annotation["bbox_coordinates"]
        labels.append((class_id, *bbox))
    return torch.Tensor(labels)


SegmentationAnalysisManager(
    ...,
    labels_extractor=labels_extractor
)
```

## Community
<table style="border: 0">
  <tr>
    <td><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/discord.png" width="60pt"></td>
    <td><a href="https://discord.gg/2v6cEGMREN"> Click here to join our Discord Community</a></td>
  </tr>
</table>

## License

This project is released under the [Apache 2.0 license](LICENSE.md).
