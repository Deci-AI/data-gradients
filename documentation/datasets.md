# Built-in Datasets

## Object Detection
### Yolo Dataset

#### Description
- Each image is associated with a label file.
- Each label file is a `.txt` where each line represents a Bounding Box in the following format:
    - `label, cx, cy, w, h` - normalized (between 0-1)

#### Example
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
from data_gradients.datasets.detection import YoloDetectionDataset

train_loader = YoloDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
val_loader = YoloDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
```
