# Built-in Datasets

## Object Detection
### Paired Image-Label Dataset

The Paired Image-Label Detection Dataset is a minimalistic and flexible Dataset class for loading datasets 
with a one-to-one correspondence between an image file and a corresponding label text file.

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
Each bounding box is represented by 5 elements.
  - 1 representing the class id
  - 4 representing the bounding box coordinates.

The class id can be at the beginning or at the end of the row, but this format needs to be consistent throughout the dataset.
Example:
  - `class_id x1 y1 x2 y2`
  - `cx, cy, w, h, class_id`
  - `class_id x, y, w, h`
  - ...

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
from data_gradients.datasets.detection import PairedImageLabelDetectionDataset

train_loader = PairedImageLabelDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
val_loader = PairedImageLabelDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
```

### XML Paired Image-Label Dataset

The XML Paired Image-Label Detection Dataset is a minimalistic and flexible Dataset class for loading datasets
with a one-to-one correspondence between an image file and a corresponding label XML file.

#### Expected folder structure
Any structure including at least one sub-directory for images and one for xml labels. They can be the same.

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
            │   ├── 1.xml
            │   ├── 2.xml
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
        │   ├── 1.xml
        │   ├── 2.jpg
        │   ├── 2.xml
        │   └── ...
        └── validation/
            ├── ...
```

**Note**: The label file need to be stored in XML format, but the file extension can be different.

#### Expected label files structure
The label files must be structured in XML format, like in the following example:

``` xml
<annotation>
    <object>
        <name>chair</name>
        <bndbox>
            <xmin>1</xmin>
            <ymin>213</ymin>
            <xmax>263</xmax>
            <ymax>375</ymax>
        </bndbox>
    </object>
    <object>
        <name>sofa</name>
        <bndbox>
            <xmin>104</xmin>
            <ymin>151</ymin>
            <xmax>334</xmax>
            <ymax>287</ymax>
        </bndbox>
    </object>
</annotation>
```


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
from data_gradients.datasets.detection import PairedImageLabelDetectionDataset

train_loader = PairedImageLabelDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
val_loader = PairedImageLabelDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
```

This class does NOT support dataset formats such as YOLO or COCO.
