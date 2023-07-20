# Built-in Datasets

DataGradients offer a few basic datasets which can help you load your data without needing to provide any additional code. 
These datasets contain only the very basic functionalities and are not recommended for training.

## Object Detection


### Yolo Format Detection Dataset

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

train_set = YoloFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
val_set = YoloFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
```

### VOC Format Detection Dataset

The VOC format Detection Dataset supports datasets where labels are stored in XML following according to VOC standard.

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
from data_gradients.datasets.detection import VOCFormatDetectionDataset

train_set = VOCFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/train", labels_dir="labels/train")
val_set = VOCFormatDetectionDataset(root_dir="<path/to/dataset_root>", images_dir="images/validation", labels_dir="labels/validation")
```


### VOC Detection Dataset
VOC Detection Dataset is a sub-class of the [VOC Format Detection Dataset](#voc_format_detection_dataset), 
where the folders are structured exactly similarly to the original PascalVOC.

#### Expected folder structure
Any structure including at least one sub-directory for images and one for xml labels. They can be the same.

Example 1: Separate directories for images and labels
```
dataset_root/
    ├── VOC2007/
    │   ├── JPEGImages/
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   └── ...
    │   ├── Annotations/
    │   │   ├── 1.xml
    │   │   ├── 2.xml
    │   │   └── ...
    │   └── ImageSets/
    │       └── Main
    │           ├── train.txt
    │           ├── val.txt
    │           ├── train_val.txt
    │           └── ...
    └── VOC2012/
        └── ...
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
Let's take an example where we only have VOC2012
```
dataset_root/
    └── VOC2012/
        ├── JPEGImages/
        │   ├── 1.jpg
        │   ├── 2.jpg
        │   └── ...
        ├── Annotations/
        │   ├── 1.xml
        │   ├── 2.xml
        │   └── ...
        └── ImageSets/
            └── Main
                ├── train.txt
                └── val.txt
```

```python
from data_gradients.datasets.detection import VOCDetectionDataset

train_set = VOCDetectionDataset(root_dir="<path/to/dataset_root>", year=2012, image_set="train")
val_set = VOCDetectionDataset(root_dir="<path/to/dataset_root>", year=2012, image_set="val")
```


### Coco Format Detection Dataset
The Coco Format Detection Dataset supports datasets where labels and annotations are stored in COCO format.

#### Expected folder structure
The dataset folder structure should include at least one sub-directory for images and one JSON file for annotations.

Example:
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
        └── annotations/
            ├── train.json
            ├── test.json
            └── validation.json
```
#### Expected Annotation File Structure
The annotation files must be structured in JSON format following the COCO data format.

#### Instantiation
```
dataset_root/
    ├── images/
    │   ├── train/
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── ...
    │   └── test/
    │       ├── ...
    └── annotations/
        ├── train.json
        ├── test.json
        └── validation.json
```

```python
from data_gradients.datasets.detection import COCOFormatDetectionDataset

train_set = COCOFormatDetectionDataset(
    root_dir="<path/to/dataset_root>", images_subdir="images/train", annotation_file_path="annotations/train.json"
)
val_set = COCOFormatDetectionDataset(
    root_dir="<path/to/dataset_root>", images_subdir="images/validation", annotation_file_path="annotations/validation.json"
)
```


### Coco Detection Dataset
Coco Detection Dataset expects the exact same annotation files and dataset structure os the original Coco dataset.

#### Expected folder structure
The dataset folder structure should 

Example:
```
dataset_root/
    ├── images/
    │   ├── train2017/
    │   ├── val2017/
    │   └── ...
    └── annotations/
        ├── instances_train2017.json
        ├── instances_val2017.json
        └── ...
```

#### Instantiation
To instantiate a dataset object for training data of the year 2017, use the following code:

```python
from data_gradients.datasets.detection import COCODetectionDataset

train_set = COCODetectionDataset(root_dir="<path/to/dataset_root>", split="train", year=2017)
val_set = COCODetectionDataset(root_dir="<path/to/dataset_root>", split="val", year=2017)
```
