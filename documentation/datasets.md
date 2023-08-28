# Built-in Datasets

DataGradients offer a few basic datasets which can help you load your data without needing to provide any additional code. 

These datasets contain only the basic functionalities. 
They are meant to be used within SuperGradients and are not recommended to be used for training (No `transform` parameter available).

## List of Datasets

- [Detection Datasets](#detection-datasets)
    - [1. COCODetectionDataset](#1-cocodetectiondataset)
    - [2. COCOFormatDetectionDataset](#2-cocoformatdetectiondataset)
    - [3. VOCDetectionDataset](#3-vocdetectiondataset)
    - [4. VOCFormatDetectionDataset](#4-vocformatdetectiondataset)
    - [5. YoloFormatDetectionDataset](#5-yoloformatdetectiondataset)
- [Segmentation Datasets](#segmentation-datasets)
    - [1. COCOFormatSegmentationDataset](#1-cocoformatsegmentationdataset)
    - [2. COCOSegmentationDataset](#2-cocosegmentationdataset)
    - [3. VOCFormatSegmentationDataset](#3-vocformatsegmentationdataset)
    - [4. VOCSegmentationDataset](#4-vocsegmentationdataset)


## Detection Datasets

<br/>

### 1. COCODetectionDataset

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


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/detection/coco_detection_dataset.py)*

<br/>

### 2. COCOFormatDetectionDataset

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


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/detection/coco_format_detection_dataset.py)*

<br/>

### 3. VOCDetectionDataset

VOC Detection Dataset is a sub-class of the VOCFormatDetectionDataset,
but where the folders are structured exactly similarly to the original PascalVOC.

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

train_set = VOCDetectionDataset(root_dir="<path/to/dataset_root>", year=2012, split="train")
val_set = VOCDetectionDataset(root_dir="<path/to/dataset_root>", year=2012, split="val")
```


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/detection/voc_detection_dataset.py)*

<br/>

### 4. VOCFormatDetectionDataset

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
        ├── train.txt
        ├── validation.txt
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

The (optional) config file should include the list image ids to include.
```
1
5
6
...
34122
```
The associated images/labels will then be loaded from the images_subdir and labels_subdir.
If config_path is not provided, all images will be used.

#### Instantiation
```
dataset_root/
    ├── train.txt
    ├── validation.txt
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

train_set = VOCFormatDetectionDataset(
    root_dir="<path/to/dataset_root>", images_subdir="images/train", labels_subdir="labels/train", config_path="train.txt"
)
val_set = VOCFormatDetectionDataset(
    root_dir="<path/to/dataset_root>", images_subdir="images/validation", labels_subdir="labels/validation", config_path="validation.txt"
)
```


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/detection/voc_format_detection_dataset.py)*

<br/>

### 5. YoloFormatDetectionDataset

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
The label files must be structured such that each row represents a bounding box label.
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

This class does NOT support dataset formats such as Pascal VOC or COCO.


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/detection/yolo_format_detection_dataset.py)*

<br/>


## Segmentation Datasets

<br/>

### 1. COCOFormatSegmentationDataset

The Coco Format Segmentation Dataset supports datasets where labels and masks are stored in COCO format.

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
The annotation files must be structured in JSON format following the COCO data format, including mask data.

#### Instantiation
```python
from data_gradients.datasets.segmentation import COCOFormatSegmentationDataset
train_set = COCOFormatSegmentationDataset(
    root_dir="<path/to/dataset_root>",
    images_subdir="images/train",
    annotation_file_path="annotations/train.json"
)
val_set = COCOFormatSegmentationDataset(
    root_dir="<path/to/dataset_root>",
    images_subdir="images/validation",
    annotation_file_path="annotations/validation.json"
)
```


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/segmentation/coco_format_segmentation_dataset.py)*

<br/>

### 2. COCOSegmentationDataset

The COCOSegmentationDataset class is a convenience subclass of the COCOFormatSegmentationDataset that simplifies
the instantiation for the widely-used COCO Segmentation Dataset.

This class assumes the default COCO dataset structure and naming conventions. The data should be stored in a specific
structure where each split of data (train, val) and year of the dataset is kept in a different directory.

#### Expected folder structure

```
dataset_root/
    ├── images/
    │   ├── train2017/
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   └── ...
    │   └── val2017/
    │       ├── 15481.jpg
    │       ├── 15482.jpg
    │       └── ...
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

#### Instantiation

```python
from data_gradients.datasets.segmentation import COCOSegmentationDataset
train_set = COCOSegmentationDataset(root_dir="<path/to/dataset_root>", split="train", year=2017)
val_set = COCOSegmentationDataset(root_dir="<path/to/dataset_root>", split="val", year=2017)
```


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/segmentation/coco_segmentation_dataset.py)*

<br/>

### 3. VOCFormatSegmentationDataset

The VOC format Segmentation Dataset supports datasets where labels are stored as images, with each color in the image representing a different class.

#### Expected folder structure
Similar to the VOCFormatDetectionDataset, this class also expects certain folder structures. For example:

Example: Separate directories for images and labels
```
    dataset_root/
        ├── train.txt
        ├── validation.txt
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
            │   ├── 1.png
            │   ├── 2.png
            │   └── ...
            ├── test/
            │   ├── ...
            └── validation/
                ├── ...
```
Each label image should be a color image where the color of each pixel corresponds to the class of that pixel.

The (optional) config file should include the list image ids to include.
```
1
5
6
# And so on ...
```
The associated images/labels will then be loaded from the images_subdir and labels_subdir.
If config_path is not provided, all images will be used.

#### Instantiation
```
from data_gradients.datasets.segmentation import VOCFormatSegmentationDataset

color_map = [
    [0, 0, 0],      # class 0
    [255, 0, 0],    # class 1
    [0, 255, 0],    # class 2
    # ...
]

train_set = VOCFormatSegmentationDataset(
    root_dir="<path/to/dataset_root>",
    images_subdir="images/train",
    labels_subdir="labels/train",
    class_names=["background", "class1", "class2"],
    color_map=color_map,
    config_path="train.txt"
)
val_set = VOCFormatSegmentationDataset(
    root_dir="<path/to/dataset_root>",
    images_subdir="images/validation",
    labels_subdir="labels/validation",
    class_names=["background", "class1", "class2"],
    color_map=color_map,
    config_path="validation.txt"
)
```


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/segmentation/voc_format_segmentation_dataset.py)*

<br/>

### 4. VOCSegmentationDataset


The VOCSegmentationDataset is specifically tailored for loading PASCAL VOC segmentation datasets.

#### Expected folder structure
Similar to the VOCFormatSegmentationDataset, this class also expects certain folder structures.
The folder structure of the PASCAL VOC dataset is as follows:

```
    dataset_root/
        ├── VOC2007/
        │   ├── JPEGImages/
        │   ├── SegmentationClass/
        │   └── ImageSets/
        │       └── Segmentation/
        │           ├── train.txt
        │           └── val.txt
        └── VOC2012/
            ├── JPEGImages/
            ├── SegmentationClass/
            └── ImageSets/
                └── Segmentation/
                    ├── train.txt
                    └── val.txt
```
Each label image should be a color image where the color of each pixel corresponds to the class of that pixel.

#### Instantiation
```
from data_gradients.datasets.segmentation import VOCSegmentationDataset

train_set = VOCSegmentationDataset(
    root_dir="<path/to/dataset_root>",
    year=2007,
    split="train",
    verbose=True
)
val_set = VOCSegmentationDataset(
    root_dir="<path/to/dataset_root>",
    year=2007,
    split="val",
    verbose=True
)
```


*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/datasets/segmentation/voc_segmentation_dataset.py)*

<br/>
