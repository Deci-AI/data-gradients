# Dataset Extractors in DataGradients

**If your dataset isn't plug-and-play with DataGradients, Dataset Extractors are here to help!**

## Table of Contents
1. [Introduction](#1-introduction)
2. [What are Dataset Extractors?](#2-what-are-dataset-extractors)
3. [When Do You Need Dataset Extractors?](#3-when-do-you-need-dataset-extractors)
4. [Implementing Dataset Extractors](#4-implementing-dataset-extractors)
5. [Extractor Structures](#5-extractor-structures)
   - [Image Extractor](#image-extractor)
   - [Label Extractor](#label-extractor)
6. [Practical Example](#6-practical-example)


## 1. Introduction
DataGradients aims to automatically recognize your dataset's structure and output format. 
This includes variations in image channel order, bounding box format, and segmentation mask type. 

However, unique datasets, especially with a nested data structure, may require Dataset Extractors for customized handling.


## 2. What are Dataset Extractors?
Dataset Extractors are user-defined functions that guide DataGradients in interpreting non-standard datasets. 
The two primary extractors are:
- **`images_extractor`**: Responsible for extracting image data in a friendly format.
- **`labels_extractor`**: Responsible for extracting label data in a friendly format.


## 3. When Do You Need Dataset Extractors?
DataGradients is designed to automatically recognize standard dataset structures. 
Yet, intricate or nested formats might be challenging for auto-inference. 

For these unique datasets, Dataset Extractors ensure seamless interfacing with DataGradients.


## 4. Implementing Dataset Extractors
After determining the need for extractors, integrate them during the instantiation of the Analysis Manager. 
For illustration:

```python
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager

# Sample dataset returns: (image, {"masks", "bboxes"})
images_extractor = lambda data: data[0]  # Extract the image
labels_extractor = lambda data: data[1]['masks']  # Extract the masks

SegmentationAnalysisManager(
    report_title="Test with Extractors",
    train_data=train_data,
    val_data=val_data,
    images_extractor=images_extractor, 
    labels_extractor=labels_extractor
)
```

## 5. Extractor Structures

### Image Extractor
Function signature:
```python
images_extractor(data: Any) -> torch.Tensor
```
Output must be a tensor representing your image(s):
  - Batched: `(BS, C, H, W)`, `(BS, H, W, C)`, `(BS, H, W)`
  - Single Image: `(C, H, W)`, `(H, W, C)`, `(H, W)`
  - Where:
    - `C`: Number of channels (e.g., 3 for RGB)
    - `BS`: Batch Size
    - `H`, `W`: Height and Width, respectively

### Label Extractor
Function signature:
```python
labels_extractor(data: Any) -> torch.Tensor
```
Depending on the task, the tensor format will differ:

- **Segmentation**:
  - Batched: `(BS, C, H, W)`, `(BS, H, W, C)`, `(BS, H, W)`
  - Single Image: `(C, H, W)`, `(H, W, C)`, `(H, W)`
- **Detection**:
  - Batched: `(BS, N, 5)`, `(N, 6)`
  - Single Image: `(N, 5)`
  - Last dimension details: `class_id, x1, y1, x2, y2`
- Where:
  - `C`: Number of channels (e.g., 3 for RGB)
  - `BS`: Batch Size
  - `H`, `W`: Height and Width, respectively

## 6. Practical Example
For a dataset returning a tuple `(image, annotation)` where `annotation` is structured as follows:

```python
annotation = [
    {"bbox_coordinates": [1.08, 187.69, 611.59, 285.84], "class_id": 51},
    ...
]
```

A suitable `labels_extractor` would be:

```python
import torch

def labels_extractor(data) -> torch.Tensor:
    _, annotations = data # annotations = [{"bbox_coordinates": [1.08, 187.69, 611.59, 285.84], "class_id": 51}, ...]
    labels = []
    for annotation in annotations:
        class_id = annotation["class_id"]
        bbox = annotation["bbox_coordinates"]
        labels.append((class_id, *bbox))
    return torch.Tensor(labels) # np.array([[51, 1.08, 187.69, 611.59, 285.84], ...])
```
