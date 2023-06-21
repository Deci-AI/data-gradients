# Features

## List of Features

- [Image](#image)
    - [Color Distribution](#0.-color-distribution)
    - [Image Brightness Distribution](#1.-image-brightness-distribution)
    - [Image Width and Height Distribution](#2.-image-width-and-height-distribution)
    - [General Statistics](#3.-general-statistics)
- [Object Detection](#object-detection)
    - [Distribution of Bounding Box Area](#0.-distribution-of-bounding-box-area)
    - [Intersection of Bounding Boxes](#1.-intersection-of-bounding-boxes)
    - [Distribution of Bounding Box per image](#2.-distribution-of-bounding-box-per-image)
    - [Distribution of Bounding Box Width and Height](#3.-distribution-of-bounding-box-width-and-height)
    - [Class Frequency](#4.-class-frequency)
    - [Bounding Boxes Density](#5.-bounding-boxes-density)
    - [Distribution of Class Frequency per Image](#6.-distribution-of-class-frequency-per-image)
    - [Visualization of Samples](#7.-visualization-of-samples)
- [Segmentation](#segmentation)
    - [Distribution of Object Area](#0.-distribution-of-object-area)
    - [Distribution of Object Width and Height](#1.-distribution-of-object-width-and-height)
    - [Class Frequency](#2.-class-frequency)
    - [Objects Density](#3.-objects-density)
    - [Distribution of Class Frequency per Image](#4.-distribution-of-class-frequency-per-image)
    - [Objects Convexity](#5.-objects-convexity)
    - [Objects Stability to Erosion](#6.-objects-stability-to-erosion)
    - [Distribution of Objects per Image](#7.-distribution-of-objects-per-image)
    - [Visualization of Samples](#8.-visualization-of-samples)


## Features Descriptions

### Image

#### 0. Color Distribution

Here's a comparison of RGB or grayscale intensity intensity (0-255) distributions across the entire dataset, assuming RGB channel ordering. 
It can reveal discrepancies in the image characteristics between the two datasets, as well as potential flaws in the augmentation process. 
E.g., a notable difference in the mean value of a specific color between the two datasets may indicate an issue with augmentation.

#### 1. Image Brightness Distribution

This graph shows the distribution of the image brightness of each dataset. 
This may for instance uncover differences between the training and validation sets, such as the presence of exclusively daytime images in the training set and nighttime images in the validation set.

#### 2. Image Width and Height Distribution

These histograms depict the distributions of image height and width. It's important to note that if certain images have been rescaled or padded, the histograms will represent the size after the rescaling and padding operations.

#### 3. General Statistics

<table align="center" border="0" cellpadding="1" cellspacing="1" style="width:800px">
    <thead>
    <tr>
        <th scope="col" style="column-width: 300px;">
            <h2>&nbsp;</h2>
        </th>
        <th scope="col" class="train_header">
            <strong>Train</strong>
        </th>
        <th scope="col" class="val_header">
            <strong>Validation</strong>
        </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td style="text-align:left; color:black;">Images</td>
        <td class="train_header"><strong>0</strong></td>
        <td class="val_header"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Classes</td>
        <td class="train_header"><strong>0</strong></td>
        <td class="val_header"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Classes in use</td>
        <td class="train_header"><strong>0</strong></td>
        <td class="val_header"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Annotations</td>
        <td class="train_header"><strong>0</strong></td>
        <td class="val_header"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Annotations per images</td>
        <td class="train_text"><strong>[]</strong></td>
        <td class="val_text"><strong>[]</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Images with no annotations</td>
        <td class="train_text"><strong>0</strong></td>
        <td class="val_text"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Median image resolution</td>
        <td class="train_text"><strong>0</strong></td>
        <td class="val_text"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Smallest annotation</td>
        <td class="train_text"><strong>0</strong></td>
        <td class="val_text"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Largest annotation</td>
        <td class="train_text"><strong>0</strong></td>
        <td class="val_text"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Most annotations in an image</td>
        <td class="train_text"><strong>0</strong></td>
        <td class="val_text"><strong>0</strong></td>
    </tr>
    <tr>
        <td style="text-align:left; color:black;">Least annotations in an image</td>
        <td class="train_text"><strong>0</strong></td>
        <td class="val_text"><strong>0</strong></td>
    </tr>

    </tbody>
</table>


### Object Detection

#### 0. Distribution of Bounding Box Area

This graph shows the distribution of bounding box area for each class. This can highlight distribution gap in object size between the training and validation splits, which can harm the model performance. 
Another thing to keep in mind is that having too many very small objects may indicate that your are down sizing your original image to a low resolution that is not appropriate for your objects.

#### 1. Intersection of Bounding Boxes

The distribution of the box Intersection over Union (IoU) with respect to other boxes in the sample. The heatmap shows the percentage of boxes overlap with IoU in range [0..T] for each class. Only intersection of boxes of same class are considered.

#### 2. Distribution of Bounding Box per image

These graphs shows how many bounding boxes appear in images. 
This can typically be valuable to know when you observe a very high number of bounding boxes per image, as some models include a parameter to filter the top k results.

#### 3. Distribution of Bounding Box Width and Height

These heat maps illustrate the distribution of bounding box width and height per class. 
Large variations in object size can affect the model's ability to accurately recognize objects.

#### 4. Class Frequency

Frequency of appearance of each class. This may highlight class distribution gap between training and validation splits. 
For instance, if one of the class only appears in the validation set, you know in advance that your model won't be able to learn to predict that class.

#### 5. Bounding Boxes Density

The heatmap represents areas of high object density within the images, providing insights into the spatial distribution of objects. By examining the heatmap, you can quickly identify if objects are predominantly concentrated in specific regions or if they are evenly distributed throughout the scene. This information can serve as a heuristic to assess if the objects are positioned appropriately within the expected areas of interest.

#### 6. Distribution of Class Frequency per Image

This graph shows how many times each class appears in an image. It highlights whether each class has a constant number of appearance per image, or whether it really depends from an image to another.

#### 7. Visualization of Samples

The sample visualization feature provides a visual representation of images and labels. This visualization aids in understanding of the composition of the dataset.


### Segmentation

#### 0. Distribution of Object Area

This graph shows the distribution of object area for each class. This can highlight distribution gap in object size between the training and validation splits, which can harm the model performance. 
Another thing to keep in mind is that having too many very small objects may indicate that your are down sizing your original image to a low resolution that is not appropriate for your objects.

#### 1. Distribution of Object Width and Height

These heat maps illustrate the distribution of objects width and height per class. 
Large variations in object size can affect the model's ability to accurately recognize objects.

#### 2. Class Frequency

This bar plot represents the frequency of appearance of each class. This may highlight class distribution gap between training and validation splits. For instance, if one of the class only appears in the validation set, you know in advance that your model won't be able to learn to predict that class.

#### 3. Objects Density

The heatmap represents areas of high object density within the images, providing insights into the spatial distribution of objects. By examining the heatmap, you can quickly identify if objects are predominantly concentrated in specific regions or if they are evenly distributed throughout the scene. This information can serve as a heuristic to assess if the objects are positioned appropriately within the expected areas of interest.

#### 4. Distribution of Class Frequency per Image

This graph shows how many times each class appears in an image. It highlights whether each class has a constant number of appearance per image, or whether it really depends from an image to another.

#### 5. Objects Convexity

This graph depicts the convexity distribution of objects in both training and validation sets. 
Higher convexity values suggest complex structures that may pose challenges for accurate segmentation.

#### 6. Objects Stability to Erosion

Assessment of object stability under morphological opening - erosion followed by dilation. When a lot of components are small then the number of components decrease which means we might have noise in our annotations (i.e 'sprinkles').

#### 7. Distribution of Objects per Image

These graphs shows how many different objects appear in images. 
This can typically be valuable to know when you observe a very high number of objects per image, as some models include a parameter to filter the top k results.

#### 8. Visualization of Samples

The sample visualization feature provides a visual representation of images and labels. This visualization aids in understanding of the composition of the dataset.
