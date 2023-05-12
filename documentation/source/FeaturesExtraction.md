# Features Extraction

During iteration across the dataset, Data Gradients computes a set of features for each sample.
Depending on the sample type, different features are computed. However, there is a set of features that are computed for all sample types.

Here's a non-exhaustive list of features that are computed for each sample type. This should give you an idea of what kind of information we extract from the dataset:

## Common Features

- Image size, area and aspect ratio
- Min, Average and max brightness
- Per-channel mean and standard deviation

Full list of image features can be found in `data_gradients.feature_extractors.features.ImageFeatures` enum.

## Segmentation Features

- Class distribution
- Segmentation mask area
- How sparse & solid mask is
- Where it is located in the image

Full list of segmentation features can be found in `data_gradients.feature_extractors.features.SegmentationMaskFeatures` enum.


All computed features are stored in a `data_gradients.feature_extractors.FeaturesCollection`. 
This object becomes a final result of the feature extraction step and is passed to the next step - [report generation](Reporting.md).
