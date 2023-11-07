## Features Description

This page focuses on the description of features.

If you are interested in using these features, there is a tutorial specifically about [Features Configuration](feature_configuration.md).

### List of Features

- [Image Features](#image-features)
    - [1. ImageColorDistribution](#1-imagecolordistribution)
    - [2. ImageDuplicates](#2-imageduplicates)
    - [3. ImagesAverageBrightness](#3-imagesaveragebrightness)
    - [4. ImagesResolution](#4-imagesresolution)
- [Object Detection Features](#object-detection-features)
    - [1. DetectionBoundingBoxArea](#1-detectionboundingboxarea)
    - [2. DetectionBoundingBoxIoU](#2-detectionboundingboxiou)
    - [3. DetectionBoundingBoxPerImageCount](#3-detectionboundingboxperimagecount)
    - [4. DetectionBoundingBoxSize](#4-detectionboundingboxsize)
    - [5. DetectionClassFrequency](#5-detectionclassfrequency)
    - [6. DetectionClassHeatmap](#6-detectionclassheatmap)
    - [7. DetectionClassesPerImageCount](#7-detectionclassesperimagecount)
    - [8. DetectionResizeImpact](#8-detectionresizeimpact)
    - [9. DetectionSampleVisualization](#9-detectionsamplevisualization)
- [Segmentation Features](#segmentation-features)
    - [1. SegmentationBoundingBoxArea](#1-segmentationboundingboxarea)
    - [2. SegmentationBoundingBoxResolution](#2-segmentationboundingboxresolution)
    - [3. SegmentationClassFrequency](#3-segmentationclassfrequency)
    - [4. SegmentationClassHeatmap](#4-segmentationclassheatmap)
    - [5. SegmentationClassesPerImageCount](#5-segmentationclassesperimagecount)
    - [6. SegmentationComponentsConvexity](#6-segmentationcomponentsconvexity)
    - [7. SegmentationComponentsErosion](#7-segmentationcomponentserosion)
    - [8. SegmentationComponentsPerImageCount](#8-segmentationcomponentsperimagecount)
    - [9. SegmentationSampleVisualization](#9-segmentationsamplevisualization)
- [Classification Features](#classification-features)
    - [1. ClassificationClassDistributionVsArea](#1-classificationclassdistributionvsarea)
    - [2. ClassificationClassDistributionVsAreaPlot](#2-classificationclassdistributionvsareaplot)
    - [3. ClassificationClassFrequency](#3-classificationclassfrequency)
    - [4. ClassificationSummaryStats](#4-classificationsummarystats)


### Image Features

<br/>

#### 1. ImageColorDistribution

Analyzes and presents the color intensity distribution across image datasets.

This feature assesses the distribution of color intensities in images and provides detailed visualizations for each
color channel. It is designed to highlight differences and consistencies in color usage between training and
validation datasets, which can be critical for adjusting image preprocessing parameters or for enhancing data augmentation techniques.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/common/image_color_distribution.py)*

<br/>

#### 2. ImageDuplicates

Extracts image duplicates, in the directories train_image_dir, valid_image_dir (when present) and their intersection.

Under the hood, uses Difference Hashing (http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)
 and considers duplicates if and only if they have the exact same hash code. This means that regardless of color format
  (i.e BGR, RGB greyscale) duplicates will be found, but might result (rarely) in false positives.

Attributes:
    train_image_dir: str, The directory containing all train images. When None, will ask the user using prompt for input.

    valid_image_dir: str. Ignored when val_data of the AbstractManager is None. The directory containing all
     valid images. When None, will ask the user using prompt for input.


    The following attributes are populated after calling self.aggreagate():

        self.train_dups: List[List[str]], a list of all image duplicate paths inside train_image_dir.

        self.valid_dups: List[List[str]], a list of all image duplicate paths inside valid_image_dir.

        self.intersection_dups: List[List[str]], a list of all image duplicate paths, that are duplicated in
         valid_image_dir and train_image_dir (i.e images that appear in train and validation).

        train_dups_appearences: int, total image count of duplicated images in train_image_dir.

        validation_dups_appearences, int, total image count of duplicated images in valid_image_dir.

        intersection_train_appearnces, int, total image count in train_image_dir that appear in valid_image_dir.

        intersection_val_appearnces int, total image count in valid_image_dir that appear in train_image_dir.


    Example:
        After running self.aggreagte() on COCO2017 detection dataset (train_image_dir="/data/coco/images/train2017/",
         valid_image_dir=/data/coco/images/val2017/):

        self.train_dups: [['/data/coco/images/train2017/000000216265.jpg', '/data/coco/images/train2017/000000411274.jpg']...]
        self.valid_dups: [] -> no duplicates in validation
        self.intersection_dups: [['/data/coco/images/train2017/000000080010.jpg', '/data/coco/images/val2017/000000140556.jpg'],
                                    ['/data/coco/images/train2017/000000535889.jpg', '/data/coco/images/val2017/000000465129.jpg']]

        self.train_dups_appearences: 72
        self.validation_dups_appearences: 0
        self.intersection_train_appearnces: 2
        self.intersection_val_appearnces: 2

        IMPORTANT: We get len(self_train_dups) = 35, but there are 72 appearences pf duplicated images in the train directory.
            This is because we have two triplet duplicates inside our train data.


NOTES:
     - DOES NOT TAKE IN ACCOUNT ANY DATASET INTERNAL LOGIC, SIMPLY EXTRACTS THE DUPLICATES FROM THESE DIRECTORIES.
     - If an image in the image directory can't be loaded, no duplicates are searched for the image.
     - Supported image formats: 'JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF', 'GIF', 'SVG', 'PGM', 'PBM', 'WEBP'.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/common/image_duplicates.py)*

<br/>

#### 3. ImagesAverageBrightness

Provides a graphical representation of image brightness distribution.

This feature quantifies the brightness of images and plots the distribution per data split, aiding in the detection of
variances like uniform lighting conditions. Useful for comparing training and validation sets to ensure model robustness
against varying brightness levels.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/common/image_average_brightness.py)*

<br/>

#### 4. ImagesResolution

Analyzes the distribution of image dimensions within a dataset.

This feature extractor records and summarizes the height and width of images, highlighting the range and commonality of different resolutions.
This analysis is beneficial for understanding the dataset’s composition and preparing for any necessary image preprocessing.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/common/image_resolution.py)*

<br/>


### Object Detection Features

<br/>

#### 1. DetectionBoundingBoxArea

Analyzes and visualizes the size distribution of objects across dataset splits.

This feature extractor calculates the area occupied by objects in images and displays a comparison
across different dataset splits. It helps in understanding the diversity in object sizes within the dataset
and flags potential disparities between training and validation sets that could impact model performance.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/bounding_boxes_area.py)*

<br/>

#### 2. DetectionBoundingBoxIoU

Computes the pairwise Intersection over Union (IoU) for bounding boxes within each image to
identify potential duplicate or highly overlapping annotations.

The computed IoU can be aggregated across classes (class-agnostic) or within the same class,
providing insights into annotation quality and potential issues with overlapping objects.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/bounding_boxes_iou.py)*

<br/>

#### 3. DetectionBoundingBoxPerImageCount

Feature Extractor to count the number of Bounding Boxes per Image.

It compiles the bounding box counts into a histogram distribution, allowing for easy identification
of the frequency of bounding box occurrences across images in a dataset.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/bounding_boxes_per_image_count.py)*

<br/>

#### 4. DetectionBoundingBoxSize

Feature Extractor to gather and analyze the relative size of Bounding Boxes within images.

It computes each bounding box's width and height as a percentage of the image's width and height,
respectively, allowing for a scale-invariant analysis of object sizes.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/bounding_boxes_resolution.py)*

<br/>

#### 5. DetectionClassFrequency

Analyzes and visualizes the distribution of class instances across dataset splits.

This feature extractor quantifies the frequency of each class's occurrence in the dataset,
providing a visual comparison between training and validation splits. Such analysis can
reveal class imbalances that may necessitate rebalancing techniques or inform the necessity
of targeted data collection to enhance model robustness and prevent overfitting or underfitting
to particular classes.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/classes_frequency.py)*

<br/>

#### 6. DetectionClassHeatmap

Provides a visual representation of object distribution across images in the dataset using heatmaps.

It helps identify common areas where objects are frequently detected, allowing insights into potential
biases in object placement or dataset collection.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/classes_heatmap_per_class.py)*

<br/>

#### 7. DetectionClassesPerImageCount

Evaluates and illustrates the frequency of class instances within individual images.

By showing the number of times classes are seen in each image, this feature helps identify which classes are common or rare in a typical image.

This provides information such as "The class 'Human' usually appears 2 to 20 times per image".

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/classes_frequency_per_image.py)*

<br/>

#### 8. DetectionResizeImpact

Examines the influence of image resizing on bounding box visibility within datasets.

By assessing changes in bounding box sizes at various image dimensions, this feature quantifies the ratio of bounding boxes that would become smaller than
predefined size thresholds.
This analysis is crucial for determining resizing practices that prevent the loss of objects during image preprocessing.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/resize_impact.py)*

<br/>

#### 9. DetectionSampleVisualization

Constructs a visual grid of image samples from different dataset splits.

This feature assembles a grid layout to visually compare groups of images, sorted by their respective dataset splits.
It's designed to help users quickly identify and assess variations in sample distribution.
The visualization is configurable in terms of the number of images per row and column, as well as the orientation of split grouping.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/sample_visualization.py)*

<br/>


### Segmentation Features

<br/>

#### 1. SegmentationBoundingBoxArea

Visualizes the distribution of object bounding box areas in segmentation tasks.

This extractor analyzes bounding box sizes relative to the image area, revealing insights about the object size distribution across different dataset splits. It helps to identify potential size biases and supports better model generalization by ensuring a balanced representation of object scales.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/bounding_boxes_area.py)*

<br/>

#### 2. SegmentationBoundingBoxResolution

Analyzes the scale variation of object dimensions across the dataset.

This extractor calculates the height and width of objects as a percentage of the image's total height and width, respectively.
This approach provides a scale-invariant analysis of object dimensions, facilitating an understanding of the diversity in object size and
aspect ratio within the dataset, regardless of the original image dimensions.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/bounding_boxes_resolution.py)*

<br/>

#### 3. SegmentationClassFrequency

Analyzes and visualizes the distribution of class instances across dataset splits.

This feature extractor quantifies the frequency of each class's occurrence in the dataset,
providing a visual comparison between training and validation splits. Such analysis can
reveal class imbalances that may necessitate rebalancing techniques or inform the necessity
of targeted data collection to enhance model robustness and prevent overfitting or underfitting
to particular classes.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/classes_frequency.py)*

<br/>

#### 4. SegmentationClassHeatmap

Provides a visual representation of object distribution across images in the dataset using heatmaps.

It helps identify common areas where objects are frequently detected, allowing insights into potential
biases in object placement or dataset collection.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/classes_heatmap_per_class.py)*

<br/>

#### 5. SegmentationClassesPerImageCount

Evaluates and illustrates the frequency of class instances within individual images.

By showing the number of times classes are seen in each image, this feature helps identify which classes are common or rare in a typical image.

This provides information such as "The class 'Human' usually appears 2 to 20 times per image".

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/classes_frequency_per_image.py)*

<br/>

#### 6. SegmentationComponentsConvexity

Assesses the convexity of segmented objects within images of a dataset and presents the distribution across dataset splits.

Higher convexity values suggest complex structures that may pose challenges for accurate segmentation.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/components_convexity.py)*

<br/>

#### 7. SegmentationComponentsErosion

Analyzes the impact of morphological operations on the segmentation mask components within a dataset, quantifying the change in
the number of components post-erosion.

This feature useful for identifying and quantifying noise or small artifacts ('sprinkles') in segmentation masks,
which may otherwise affect the performance of segmentation models.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/components_erosion.py)*

<br/>

#### 8. SegmentationComponentsPerImageCount

Calculates and visualizes the number of distinct segmented components per image across different dataset splits.

This feature extractor counts the total number of segmented components (objects) in each image, which can provide insights into the complexity of the scenes within the dataset. It can help identify if there is a balance or imbalance in the number of objects per image across the training and validation sets. Understanding this distribution is important for adjusting model hyperparameters that may depend on the expected number of objects in a scene, such as Non-Max Suppression (NMS) thresholds or maximum detections per image.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/component_frequency_per_image.py)*

<br/>

#### 9. SegmentationSampleVisualization

Constructs a visual grid of image samples from different dataset splits.

This feature assembles a grid layout to visually compare groups of images, sorted by their respective dataset splits.
It's designed to help users quickly identify and assess variations in sample distribution.
The visualization is configurable in terms of the number of images per row and column, as well as the orientation of split grouping.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/sample_visualization.py)*

<br/>


### Classification Features

<br/>

#### 1. ClassificationClassDistributionVsArea

Summarizes how average image dimensions vary among classes and data splits.

This feature extractor calculates the mean image size (width and height) for each label within the provided splits of the dataset.
It highlights potential discrepancies in image resolutions across different classes and dataset splits, which could impact model performance.
Disparities in image sizes could indicate a need for more uniform data collection or preprocessing to avoid model biases and ensure consistent
performance across all classes and splits.

Key Uses:

- Pinpointing classes with significant variations in image resolution to inform data collection and preprocessing.
- Assessing the consistency of image resolutions across dataset splits to guide training strategies and augmentation techniques.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/classification/class_distribution_vs_area.py)*

<br/>

#### 2. ClassificationClassDistributionVsAreaPlot

Visualizes the spread of image widths and heights within each class and across data splits.

This feature extractor creates a scatter plot to graphically represent the diversity of image dimensions associated with each class label and split
in the dataset.
By visualizing this data, users can quickly assess whether certain classes or splits contain images that are consistently larger or smaller than others,
potentially indicating a need for data preprocessing or augmentation strategies to ensure model robustness.

Key Uses:

- Identifying classes with notably different average image sizes that may influence model training.
- Detecting splits in the dataset where image size distribution is uneven, prompting the need for more careful split strategies or
tailored data augmentation.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/classification/class_distribution_vs_area_scatter.py)*

<br/>

#### 3. ClassificationClassFrequency

Analyzes and visualizes the frequency of each class label across different dataset splits.

This feature extractor computes the frequency of occurrence for each class label in the dataset, providing insights into the
balance or imbalance of class distribution across training and validation.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/classification/class_frequency.py)*

<br/>

#### 4. ClassificationSummaryStats

Gathers basic statistical data from the dataset.

This extractor compiles essential statistics from the image samples. It counts the number of images, annotations, and classes,
assesses the diversity of image resolutions, and measures the size of annotations. This data is crucial for getting a high-level
overview of the dataset's characteristics and composition.

*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/classification/summary.py)*

<br/>
