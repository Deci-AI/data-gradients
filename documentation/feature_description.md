## Features Description

This page focuses on the description of features.

If you are interested in using these features, there is a tutorial specifically about [Features Configuration](feature_configuration.md).

### List of Features

- [Image Features](#image-features)
    - [1. Color Distribution](#1-color-distribution)
    - [2. Image Brightness Distribution](#2-image-brightness-distribution)
    - [3. Image Width and Height Distribution](#3-image-width-and-height-distribution)
- [Object Detection Features](#object-detection-features)
    - [1. Distribution of Bounding Box Area](#1-distribution-of-bounding-box-area)
    - [2. Intersection of Bounding Boxes](#2-intersection-of-bounding-boxes)
    - [3. Distribution of Bounding Box per image](#3-distribution-of-bounding-box-per-image)
    - [4. Distribution of Bounding Box Width and Height](#4-distribution-of-bounding-box-width-and-height)
    - [5. Class Frequency](#5-class-frequency)
    - [6. Bounding Boxes Density](#6-bounding-boxes-density)
    - [7. Distribution of Class Frequency per Image](#7-distribution-of-class-frequency-per-image)
    - [8. Visualization of Samples](#8-visualization-of-samples)
- [Segmentation Features](#segmentation-features)
    - [1. Distribution of Object Area](#1-distribution-of-object-area)
    - [2. Distribution of Object Width and Height](#2-distribution-of-object-width-and-height)
    - [3. Class Frequency](#3-class-frequency)
    - [4. Objects Density](#4-objects-density)
    - [5. Distribution of Class Frequency per Image](#5-distribution-of-class-frequency-per-image)
    - [6. Objects Convexity](#6-objects-convexity)
    - [7. Objects Stability to Erosion](#7-objects-stability-to-erosion)
    - [8. Distribution of Objects per Image](#8-distribution-of-objects-per-image)
    - [9. Visualization of Samples](#9-visualization-of-samples)


### Image Features

<br/>

#### 1. Color Distribution

Here's a comparison of RGB or grayscale intensity intensity (0-255) distributions across the entire dataset, assuming RGB channel ordering. 
It can reveal discrepancies in the image characteristics between the two datasets, as well as potential flaws in the augmentation process. 
E.g., a notable difference in the mean value of a specific color between the two datasets may indicate an issue with augmentation.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/common/image_color_distribution.py)*

<br/>

#### 2. Image Brightness Distribution

This graph shows the distribution of the image brightness of each dataset. 
This may for instance uncover differences between the training and validation sets, such as the presence of exclusively daytime images in the training set and nighttime images in the validation set.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/common/image_average_brightness.py)*

<br/>

#### 3. Image Width and Height Distribution

These histograms depict the distributions of image height and width. It's important to note that if certain images have been rescaled or padded, the histograms will represent the size after the rescaling and padding operations.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/common/image_resolution.py)*

<br/>


### Object Detection Features

<br/>

#### 1. Distribution of Bounding Box Area

This graph shows the distribution of bounding box area for each class. This can highlight distribution gap in object size between the training and validation splits, which can harm the model performance. 
Another thing to keep in mind is that having too many very small objects may indicate that your are down sizing your original image to a low resolution that is not appropriate for your objects.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/bounding_boxes_area.py)*

<br/>

#### 2. Intersection of Bounding Boxes

The distribution of the box Intersection over Union (IoU) with respect to other boxes in the sample. The heatmap shows the percentage of boxes overlap with IoU in range [0..T] for each class. Only intersection of boxes of same class are considered.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/bounding_boxes_iou.py)*

<br/>

#### 3. Distribution of Bounding Box per image

These graphs shows how many bounding boxes appear in images. 
This can typically be valuable to know when you observe a very high number of bounding boxes per image, as some models include a parameter to filter the top k results.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/bounding_boxes_per_image_count.py)*

<br/>

#### 4. Distribution of Bounding Box Width and Height

These heat maps illustrate the distribution of bounding box width and height per class. 
Large variations in object size can affect the model's ability to accurately recognize objects.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/bounding_boxes_resolution.py)*

<br/>

#### 5. Class Frequency

Frequency of appearance of each class. This may highlight class distribution gap between training and validation splits. 
For instance, if one of the class only appears in the validation set, you know in advance that your model won't be able to learn to predict that class.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/classes_frequency.py)*

<br/>

#### 6. Bounding Boxes Density

The heatmap represents areas of high object density within the images, providing insights into the spatial distribution of objects. By examining the heatmap, you can quickly identify if objects are predominantly concentrated in specific regions or if they are evenly distributed throughout the scene. This information can serve as a heuristic to assess if the objects are positioned appropriately within the expected areas of interest.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/classes_heatmap_per_class.py)*

<br/>

#### 7. Distribution of Class Frequency per Image

This graph shows how many times each class appears in an image. It highlights whether each class has a constant number of appearance per image, or whether it really depends from an image to another.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/classes_frequency_per_image.py)*

<br/>

#### 8. Visualization of Samples

The sample visualization feature provides a visual representation of images and labels. This visualization aids in understanding of the composition of the dataset.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/object_detection/sample_visualization.py)*

<br/>


### Segmentation Features

<br/>

#### 1. Distribution of Object Area

This graph shows the distribution of object area for each class. This can highlight distribution gap in object size between the training and validation splits, which can harm the model performance. 
Another thing to keep in mind is that having too many very small objects may indicate that your are down sizing your original image to a low resolution that is not appropriate for your objects.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/bounding_boxes_area.py)*

<br/>

#### 2. Distribution of Object Width and Height

These heat maps illustrate the distribution of objects width and height per class. 
Large variations in object size can affect the model's ability to accurately recognize objects.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/bounding_boxes_resolution.py)*

<br/>

#### 3. Class Frequency

This bar plot represents the frequency of appearance of each class. This may highlight class distribution gap between training and validation splits. For instance, if one of the class only appears in the validation set, you know in advance that your model won't be able to learn to predict that class.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/classes_frequency.py)*

<br/>

#### 4. Objects Density

The heatmap represents areas of high object density within the images, providing insights into the spatial distribution of objects. By examining the heatmap, you can quickly identify if objects are predominantly concentrated in specific regions or if they are evenly distributed throughout the scene. This information can serve as a heuristic to assess if the objects are positioned appropriately within the expected areas of interest.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/classes_heatmap_per_class.py)*

<br/>

#### 5. Distribution of Class Frequency per Image

This graph shows how many times each class appears in an image. It highlights whether each class has a constant number of appearance per image, or whether it really depends from an image to another.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/classes_frequency_per_image.py)*

<br/>

#### 6. Objects Convexity

This graph depicts the convexity distribution of objects in both training and validation sets. 
Higher convexity values suggest complex structures that may pose challenges for accurate segmentation.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/components_convexity.py)*

<br/>

#### 7. Objects Stability to Erosion

Assessment of object stability under morphological opening - erosion followed by dilation. When a lot of components are small then the number of components decrease which means we might have noise in our annotations (i.e 'sprinkles').
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/components_erosion.py)*

<br/>

#### 8. Distribution of Objects per Image

These graphs shows how many different objects appear in images. 
This can typically be valuable to know when you observe a very high number of objects per image, as some models include a parameter to filter the top k results.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/component_frequency_per_image.py)*

<br/>

#### 9. Visualization of Samples

The sample visualization feature provides a visual representation of images and labels. This visualization aids in understanding of the composition of the dataset.
*[source code](https://github.com/Deci-AI/data-gradients/blob/master/src/data_gradients/feature_extractors/segmentation/sample_visualization.py)*

<br/>
