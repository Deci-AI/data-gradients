# Tutorial: Finding Image Duplicates using the ImageDuplicates Feature Extractor
In this tutorial, we will explore how to utilize the ImageDuplicates class to identify duplicate images within a specified directory containing images. The ImageDuplicates class is a feature extractor that employs Difference Hashing to detect duplicate images based on their hash codes. We will guide you through the process of initializing and using this class to find duplicates within the COCO2017 detection dataset and examine the results.


## 1. Adding the ImageDuplicates to the Image Features Report Section Features List
Just like any other feature extractor, you need to include it in your feature list within the appropriate report section.
Since finding duplicates is not task-specific, it belongs in the "Image Features" report section. In your detection.yaml configuration file, add it as follows:
Note that this feature is not in the default features since it requires explicit train_image_dir, while other features don't require to change the .yaml.

````yaml
report_sections:
  - name: Image Features
    features:
      - SummaryStats
      - ImagesResolution
      - ImageColorDistribution
      - ImagesAverageBrightness
      - ImageDuplicates:
          train_image_dir: /path/to/coco/images/train2017
          valid_image_dir: /path/to/coco/images/val2017
  - name: Object Detection Features
    ...

````

Please note that train_image_dir and valid_image_dir should point to the directories containing all the training and validation images,
respectively. It's essential to understand that this feature extractor does not account for any internal dataset modifications, such as image alterations or exclusions.
It operates solely on the standard loading of images within those directories without preprocessing.

## 2. Running Analysis

Now that we have added ImageDuplicates to our feature list, running the analysis is straightforward. Here's an example:
````python

# Note: This example will require you to install the super-gradients package.
# Nevertheless, by replacing the data objects from super-gradients it would work with any dataset.
from super_gradients.training.dataloaders.dataloaders import coco2017_train, coco2017_val
from data_gradients.managers.detection_manager import DetectionAnalysisManager



train_loader = coco2017_train(dataset_params={"data_dir":"/path/to/coco/"})
val_loader = coco2017_val(dataset_params={"data_dir":"/path/to/coco/"})

analyzer = DetectionAnalysisManager(
    report_title="COCO2017DGAnalysis",
    train_data=train_loader,
    val_data=val_loader,
    class_names=train_loader.dataset.classes,
    use_cache=True,  # With this we will be asked about the dataset information only once
)

analyzer.run()
````

When performing the analysis for the detection task, you will need to answer some questions, regardless of whether you are using ImageDuplicates or not.


```
------------------------------------------------------------------------
Which comes first in your annotations, the class id or the bounding box?
------------------------------------------------------------------------
Here's a sample of how your labels look like:
Each line corresponds to a bounding box.
tensor([[ 32.0000, 269.0935, 210.8998,   7.8713,   6.2508],
        [  0.0000, 108.3852, 194.5443,  48.0178, 120.0172],
        [  0.0000,  99.6082, 189.0017,  85.9444,  97.7378],
        [  0.0000, 556.2536, 271.3510, 124.8652, 124.8652]])

Options:
[0] | Label comes first (e.g. [class_id, x1, y1, x2, y2])
[1] | Bounding box comes first (e.g. [x1, y1, x2, y2, class_id])

Your selection (Enter the corresponding number) >>> 0
0
Great! You chose: Label comes first (e.g. [class_id, x1, y1, x2, y2])


------------------------------------------------------------------------
What is the bounding box format?
------------------------------------------------------------------------
Here's a sample of how your labels look like:
Each line corresponds to a bounding box.
tensor([[ 32.0000, 269.0935, 210.8998,   7.8713,   6.2508],
        [  0.0000, 108.3852, 194.5443,  48.0178, 120.0172],
        [  0.0000,  99.6082, 189.0017,  85.9444,  97.7378],
        [  0.0000, 556.2536, 271.3510, 124.8652, 124.8652]])

Options:
[0] | xyxy: x-left, y-top, x-right, y-bottom		(Pascal-VOC format)
[1] | xywh: x-left, y-top, width, height			(COCO format)
[2] | cxcywh: x-center, y-center, width, height		(YOLO format)

Your selection (Enter the corresponding number) >>> 2
2
Great! You chose: cxcywh: x-center, y-center, width, height		(YOLO format)
```

Once the analysis is complete, you will receive the following message:


```
We have finished evaluating your dataset!

The results can be seen in:
    - /path/to/data-gradients/examples/logs/COCO2017DGAnalysis
    - /path/to/data-gradients/examples/logs/COCO2017DGAnalysis/archive_20230905-141929
```

## 3. Examine Results

After completing Step 2, your Report.pdf can be found in ```/path/to/data-gradients/examples/logs/COCO2017DGAnalysis```:


![subsection](assets/image_duplicates_subsection.png)

To maintain a concise and clean PDF report, additional information about the duplicates is stored within the JSON results file located at `/path/to/data-gradients/examples/logs/COCO2017DGAnalysis/summary.json`

Let's break down the Train duplicated images section:
In the subsection of the PDF report, it mentions that there are 35 duplicated images appearing 72 times across the dataset, implying that some images are duplicated at least 3 times.
Examining the JSON summary confirms this, revealing 33 pairs of duplicated images and 2 triplets within the train dataset. 

```json            "title": "Image Duplicates",
            "stats": {
                "Train duplicates": [
                ....
                    [
                        "/data/coco/images/train2017/000000397819.jpg",
                        "/data/coco/images/train2017/000000388662.jpg",
                        "/data/coco/images/train2017/000000578492.jpg"
                    ],
                    [
                        "/data/coco/images/train2017/000000400264.jpg",
                        "/data/coco/images/train2017/000000482201.jpg"
                    ],
                    [
                        "/data/coco/images/train2017/000000286116.jpg",
                        "/data/coco/images/train2017/000000281790.jpg"
                    ],
                    ....
                    [
                        "/data/coco/images/train2017/000000407259.jpg",
                        "/data/coco/images/train2017/000000446141.jpg",
                        "/data/coco/images/train2017/000000278029.jpg"
                    ],
                    ...
```
Fortunately, there are no duplicates within the validation set.


Lastly, let's explore the intersection of the train and validation sets:
```
                "Intersection duplicates": [
                    [
                        "/data/coco/images/train2017/000000080010.jpg",
                        "/data/coco/images/val2017/000000140556.jpg"
                    ],
                    [
                        "/data/coco/images/train2017/000000535889.jpg",
                        "/data/coco/images/val2017/000000465129.jpg"
                    ]
                ]
            
```

As stated in the report, we found 2 pairs present in the intersection.

Opening the above file paths, we indeed see we have duplicates between our train data and validation data:

| Train | Validation |
|---------|---------|
| `/data/coco/images/train2017/000000080010.jpg`<br>![subsection](assets/000000080010.jpg) | `/data/coco/images/val2017/000000140556.jpg`<br>![subsection](assets/000000080010.jpg) |
| `/data/coco/images/train2017/000000535889.jpg`<br>![subsection](assets/000000535889.jpg) | `/data/coco/images/val2017/000000465129.jpg`<br>![subsection](assets/000000465129.jpg) |



This concludes our tutorial on using the ImageDuplicates feature extractor.
