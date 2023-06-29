# Built-in Datasets

DataGradients offer a few basic datasets which can help you load your data without needing to provide any additional code. 
These datasets contain only the very basic functionalities and are not recommended for training.

## Object Detection


### Yolo Format Dataset

The Yolo format Detection Dataset supports any dataset stored in the YOLO format.

#### Expected folder structure
Any structure including at least one sub-directory for images and one for labels. They can be the same.

Example 1: Separate directories for images and labels
```
    dataset_root/
        ├── images/
        │   ├── train/
        │   │   ├── 1.jpg
        │   │   ├── 2.jpg
        │   │   └── ...
        │   ├── test/
        │   │   ├── ...
        │   └── validation/
        │       ├── ...
        └── labels/
            ├── train/
            │   ├── 1.txt
            │   ├── 2.txt
            │   └── ...
            ├── test/
            │   ├── ...
            └── validation/
                ├── ...
```

Example 2: Same directory for images and labels
```
    dataset_root/
        ├── train/
        │   ├── 1.jpg
        │   ├── 1.txt
        │   ├── 2.jpg
        │   ├── 2.txt
        │   └── ...
        └── validation/
            ├── ...
```

#### Expected label files structure
The label files must be structured such that each row represents a bounding box annotation.
Each bounding box is represented by 5 elements: `class_id, cx, cy, w, h`.

#### Instantiation
```
dataset_root/
    ├── images/
    │   ├── train/
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   └── ...
    │   ├── test/
    │   │   ├── ...
    │   └── validation/
    │       ├── ...
    └── labels/
        ├── train/
        │   ├── 1.txt
        │   ├── 2.txt
        │   └── ...
        ├── test/
        │   ├── ...
        └── validation/
            ├── ...
```

```python
from data_gradients.datasets.detection import YoloFormatDetectionDataset

train_loader = YoloFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
val_loader = YoloFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
```
