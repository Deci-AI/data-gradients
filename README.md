# DataGradients
<div align="center">
<p align="center">
  <a href="https://github.com/Deci-AI/super-gradients#prerequisites"><img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue" /></a>
  <a href="https://pypi.org/project/data-gradients/"><img src="https://img.shields.io/pypi/v/data-gradients" /></a>
  <a href="https://github.com/Deci-AI/data-gradients/releases"><img src="https://img.shields.io/github/v/release/Deci-AI/data-gradients" /></a>
  <a href="https://github.com/Deci-AI/data-gradients/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>
</p>   
</div>

DataGradients is an open-source python based library specifically designed for computer vision dataset analysis. 

It automatically extracts features from your datasets and combines them all into a single user-friendly report.

**Detect Common Data Issues** - corrupted data, labeling errors, underlying biases, data leakage, duplications, faulty augmentations, and disparities between train and validation sets. 

**Extract Insights for Better Model Design** - take informed decisions when designing your model, based on data characteristics such as 
object size and location distributions, number of object in an image and high frequency details.

**Reduce Guesswork Searching For The Right Hyperparameters** - define the correct NMS and filtering parameters, identify 
class distribution issues and define loss function weights accordingly, define proper augmentations according to data variability, 
and calibrate metrics to monitor your unique dataset.


<div style="padding: 20px; background: rgba(33,114,255,0.23) no-repeat 10px 50%; border: 1px solid #2172ff;">
    To better understand how to tackle the data issues highlighted by DataGradients, explore our comprehensive <a href="https://deci.ai/course/profiling-computer-vision-datasets-overview/?utm_campaign[…]=DG-PDF-report&utm_medium=DG-repo&utm_content=DG-Report-to-course">online course</a> on analyzing computer vision datasets.
</div>

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
   - [Prerequisites](#prerequisites)
   - [Dataset Analysis](#dataset-analysis)
   - [Report](#report)
- [Feature Configuration](#feature-configuration)
- [Dataset Extractors](#dataset-extractors)
- [Pre-computed Dataset Analysis](#pre-computed-dataset-analysis)
- [License](#license)

## Features
- Image-Level Evaluation: DataGradients evaluates key image features such as resolution, color distribution, and average brightness.
- Class Distribution: The library extracts stats allowing you to know which classes are the most used, how many objects do you have per image, how many image without any label, ...
- Heatmap Generation: DataGradients produces heatmaps of bounding boxes or masks, allowing you to understand if the objects are positioned in the right area.
- And [many more](./documentation/feature_description.md)!

<div align="center">
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_image_stats.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_image_stats.png" width="250px"></a>
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_mask_sample.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_mask_sample.png" width="250px"></a>
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_classes_distribution.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/report_classes_distribution.png" width="250px"></a>
  <p><em>Example of pages from the Report</em>
</div>

<div align="center">
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationBoundingBoxArea.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationBoundingBoxArea.png" width="375px"></a>
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationBoundingBoxResolution.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationBoundingBoxResolution.png" width="375px"></a>
  <br />
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationClassFrequency.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationClassFrequency.png" width="375px"></a>
  <a href="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationComponentsPerImageCount.png"><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/SegmentationComponentsPerImageCount.png" width="375px"></a>
  <p><em>Example of specific features</em>
</div>

## Examples 
[COCO 2017 Detection report](documentation/assets/Report_COCO.pdf)

[Cityscapes Segmentation report](documentation/assets/Report_CityScapes.pdf)

<table style="border: 0">
  <tr>
    <td><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/colab.png" width="80pt"></td>
    <td><a href="https://colab.research.google.com/drive/1dswgeK0KF-n61p6ixRdFgbQKHEtOu8SE?usp=sharing"> Example notebook on Colab</a></td>
  </tr>
</table>


## Installation
You can install DataGradients directly from the GitHub repository.

```
pip install data-gradients
```


## Quick Start

### Prerequisites

- **Dataset**: Includes a **Train** set and a **Validation** or a **Test** set.
- One of
  - **Class Names**: A list of the unique categories present in your dataset.
  - **Number of classes**: How many unique classes appear in your dataset (make sure that this number is greater than the highest class index)
- **Dataset Iterable**: A method to iterate over your Dataset providing images and labels. Can be any of the following:
  - PyTorch **Dataloader**
  - PyTorch **Dataset**
  - Generator that yields image/label pairs
  - Any other iterable you use for model training/validation

Please ensure all the points above are checked before you proceed with **DataGradients**.

**Good to Know**: DataGradients will try to find out how the dataset returns images and labels.
- If something cannot be automatically determined, you will be asked to provide some extra information through a text input.
- In some extreme cases, the process will crash and invite you to implement a custom dataset adapter (see relevant section)

**Heads up**: We currently provide a few out-of-the-box [dataset/dataloader](./documentation/datasets.md) implementation. 
You can find more dataset implementations in [PyTorch](https://pytorch.org/vision/stable/datasets.html) 
or [SuperGradients](https://docs.deci.ai/super-gradients/src/super_gradients/training/datasets/Dataset_Setup_Instructions.html). 

**Example**
``` python
from torchvision.datasets import CocoDetection

train_data = CocoDetection(...)
val_data = CocoDetection(...)
class_names = ["person", "bicycle", "car", "motorcycle", ...]
```


### Dataset Analysis
You are now ready to go, chose the relevant analyzer for your task and run it over your datasets!

**Image Classification**
```python
from data_gradients.managers.classification_manager import ClassificationAnalysisManager 

train_data = ... # Your dataset iterable (torch dataset/dataloader/...)
val_data = ... # Your dataset iterable (torch dataset/dataloader/...)
class_names = ... # [<class-1>, <class-2>, ...]

analyzer = ClassificationAnalysisManager(
    report_title="Testing Data-Gradients Classification",
    train_data=train_data,
    val_data=val_data,
    class_names=class_names,
)

analyzer.run()
```

**Object Detection**
```python
from data_gradients.managers.detection_manager import DetectionAnalysisManager

train_data = ... # Your dataset iterable (torch dataset/dataloader/...)
val_data = ... # Your dataset iterable (torch dataset/dataloader/...)
class_names = ... # [<class-1>, <class-2>, ...]

analyzer = DetectionAnalysisManager(
    report_title="Testing Data-Gradients Object Detection",
    train_data=train_data,
    val_data=val_data,
    class_names=class_names,
)

analyzer.run()
```


**Semantic Segmentation**
```python
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager 

train_data = ... # Your dataset iterable (torch dataset/dataloader/...)
val_data = ... # Your dataset iterable (torch dataset/dataloader/...)
class_names = ... # [<class-1>, <class-2>, ...]

analyzer = SegmentationAnalysisManager(
    report_title="Testing Data-Gradients Segmentation",
    train_data=train_data,
    val_data=val_data,
    class_names=class_names,
)

analyzer.run()
```

**Example**

You can test the segmentation analysis tool in the following [example](https://github.com/Deci-AI/data-gradients/blob/master/examples/segmentation_example.py)
which does not require you to download any additional data.


### Report
Once the analysis is done, the path to your pdf report will be printed.


## Feature Configuration
 
The feature configuration allows you to run the analysis on a subset of features or adjust the parameters of existing features. 
If you are interested in customizing this configuration, you can check out the [documentation](documentation/feature_configuration.md) on that topic.


## Dataset Extractors
**Ensuring Comprehensive Dataset Compatibility**

Integrating datasets with unique structures can present challenges. 
To address this, DataGradients offers `extractors` tailored for enhancing compatibility with diverse dataset formats.

**Highlights**:
- Customized dataset outputs or distinctive annotation methodologies can be seamlessly accommodated using extractors.
- DataGradients is adept at automatic dataset inference; however, certain specificities, such as distinct image channel orders or bounding box definitions, may necessitate a tailored approach.

For an in-depth understanding and implementation details, we encourage a thorough review of the [Dataset Extractors Documentation](./documentation/dataset_extractors.md).



## Pre-computed Dataset Analysis

<details>

<summary><h3>Detection</h3></summary>

Common Datasets

- [COCO](https://dgreports.deci.ai/detection/COCO/Report.pdf)

- [VOC](https://dgreports.deci.ai/detection/VOC/Report.pdf)

[Roboflow 100](https://universe.roboflow.com/roboflow-100?ref=blog.roboflow.com) Datasets

- [4-fold-defect](https://dgreports.deci.ai/detection/RF100_4-fold-defect/Report.pdf)

- [abdomen-mri](https://dgreports.deci.ai/detection/RF100_abdomen-mri/Report.pdf)

- [acl-x-ray](https://dgreports.deci.ai/detection/RF100_acl-x-ray/Report.pdf)

- [activity-diagrams-qdobr](https://dgreports.deci.ai/detection/RF100_activity-diagrams-qdobr/Report.pdf)

- [aerial-cows](https://dgreports.deci.ai/detection/RF100_aerial-cows/Report.pdf)

- [aerial-pool](https://dgreports.deci.ai/detection/RF100_aerial-pool/Report.pdf)

- [aerial-spheres](https://dgreports.deci.ai/detection/RF100_aerial-spheres/Report.pdf)

- [animals-ij5d2](https://dgreports.deci.ai/detection/RF100_animals-ij5d2/Report.pdf)

- [apex-videogame](https://dgreports.deci.ai/detection/RF100_apex-videogame/Report.pdf)

- [apples-fvpl5](https://dgreports.deci.ai/detection/RF100_apples-fvpl5/Report.pdf)

- [aquarium-qlnqy](https://dgreports.deci.ai/detection/RF100_aquarium-qlnqy/Report.pdf)

- [asbestos](https://dgreports.deci.ai/detection/RF100_asbestos/Report.pdf)

- [avatar-recognition-nuexe](https://dgreports.deci.ai/detection/RF100_avatar-recognition-nuexe/Report.pdf)

- [axial-mri](https://dgreports.deci.ai/detection/RF100_axial-mri/Report.pdf)

- [bacteria-ptywi](https://dgreports.deci.ai/detection/RF100_bacteria-ptywi/Report.pdf)

- [bccd-ouzjz](https://dgreports.deci.ai/detection/RF100_bccd-ouzjz/Report.pdf)

- [bees-jt5in](https://dgreports.deci.ai/detection/RF100_bees-jt5in/Report.pdf)

- [bone-fracture-7fylg](https://dgreports.deci.ai/detection/RF100_bone-fracture-7fylg/Report.pdf)

- [brain-tumor-m2pbp](https://dgreports.deci.ai/detection/RF100_brain-tumor-m2pbp/Report.pdf)

- [cable-damage](https://dgreports.deci.ai/detection/RF100_cable-damage/Report.pdf)

- [cables-nl42k](https://dgreports.deci.ai/detection/RF100_cables-nl42k/Report.pdf)

- [cavity-rs0uf](https://dgreports.deci.ai/detection/RF100_cavity-rs0uf/Report.pdf)

- [cell-towers](https://dgreports.deci.ai/detection/RF100_cell-towers/Report.pdf)

- [cells-uyemf](https://dgreports.deci.ai/detection/RF100_cells-uyemf/Report.pdf)

- [chess-pieces-mjzgj](https://dgreports.deci.ai/detection/RF100_chess-pieces-mjzgj/Report.pdf)

- [circuit-elements](https://dgreports.deci.ai/detection/RF100_circuit-elements/Report.pdf)

- [circuit-voltages](https://dgreports.deci.ai/detection/RF100_circuit-voltages/Report.pdf)

- [cloud-types](https://dgreports.deci.ai/detection/RF100_cloud-types/Report.pdf)

- [coins-1apki](https://dgreports.deci.ai/detection/RF100_coins-1apki/Report.pdf)

- [construction-safety-gsnvb](https://dgreports.deci.ai/detection/RF100_construction-safety-gsnvb/Report.pdf)

- [coral-lwptl](https://dgreports.deci.ai/detection/RF100_coral-lwptl/Report.pdf)

- [corrosion-bi3q3](https://dgreports.deci.ai/detection/RF100_corrosion-bi3q3/Report.pdf)

- [cotton-20xz5](https://dgreports.deci.ai/detection/RF100_cotton-20xz5/Report.pdf)

- [cotton-plant-disease](https://dgreports.deci.ai/detection/RF100_cotton-plant-disease/Report.pdf)

- [csgo-videogame](https://dgreports.deci.ai/detection/RF100_csgo-videogame/Report.pdf)

- [currency-v4f8j](https://dgreports.deci.ai/detection/RF100_currency-v4f8j/Report.pdf)

- [digits-t2eg6](https://dgreports.deci.ai/detection/RF100_digits-t2eg6/Report.pdf)

- [document-parts](https://dgreports.deci.ai/detection/RF100_document-parts/Report.pdf)

- [excavators-czvg9](https://dgreports.deci.ai/detection/RF100_excavators-czvg9/Report.pdf)

- [farcry6-videogame](https://dgreports.deci.ai/detection/RF100_farcry6-videogame/Report.pdf)

- [fish-market-ggjso](https://dgreports.deci.ai/detection/RF100_fish-market-ggjso/Report.pdf)

- [flir-camera-objects](https://dgreports.deci.ai/detection/RF100_flir-camera-objects/Report.pdf)

- [furniture-ngpea](https://dgreports.deci.ai/detection/RF100_furniture-ngpea/Report.pdf)

- [gauge-u2lwv](https://dgreports.deci.ai/detection/RF100_gauge-u2lwv/Report.pdf)

- [grass-weeds](https://dgreports.deci.ai/detection/RF100_grass-weeds/Report.pdf)

- [gynecology-mri](https://dgreports.deci.ai/detection/RF100_gynecology-mri/Report.pdf)

- [halo-infinite-angel-videogame](https://dgreports.deci.ai/detection/RF100_halo-infinite-angel-videogame/Report.pdf)

- [hand-gestures-jps7z](https://dgreports.deci.ai/detection/RF100_hand-gestures-jps7z/Report.pdf)

- [insects-mytwu](https://dgreports.deci.ai/detection/RF100_insects-mytwu/Report.pdf)

- [leaf-disease-nsdsr](https://dgreports.deci.ai/detection/RF100_leaf-disease-nsdsr/Report.pdf)

- [lettuce-pallets](https://dgreports.deci.ai/detection/RF100_lettuce-pallets/Report.pdf)

- [liver-disease](https://dgreports.deci.ai/detection/RF100_liver-disease/Report.pdf)

- [marbles](https://dgreports.deci.ai/detection/RF100_marbles/Report.pdf)

- [mask-wearing-608pr](https://dgreports.deci.ai/detection/RF100_mask-wearing-608pr/Report.pdf)

- [mitosis-gjs3g](https://dgreports.deci.ai/detection/RF100_mitosis-gjs3g/Report.pdf)

- [number-ops](https://dgreports.deci.ai/detection/RF100_number-ops/Report.pdf)

- [paper-parts](https://dgreports.deci.ai/detection/RF100_paper-parts/Report.pdf)

- [paragraphs-co84b](https://dgreports.deci.ai/detection/RF100_paragraphs-co84b/Report.pdf)

- [parasites-1s07h](https://dgreports.deci.ai/detection/RF100_parasites-1s07h/Report.pdf)

- [peanuts-sd4kf](https://dgreports.deci.ai/detection/RF100_peanuts-sd4kf/Report.pdf)

- [peixos-fish](https://dgreports.deci.ai/detection/RF100_peixos-fish/Report.pdf)

- [people-in-paintings](https://dgreports.deci.ai/detection/RF100_people-in-paintings/Report.pdf)

- [pests-2xlvx](https://dgreports.deci.ai/detection/RF100_pests-2xlvx/Report.pdf)

- [phages](https://dgreports.deci.ai/detection/RF100_phages/Report.pdf)

- [pills-sxdht](https://dgreports.deci.ai/detection/RF100_pills-sxdht/Report.pdf)

- [poker-cards-cxcvz](https://dgreports.deci.ai/detection/RF100_poker-cards-cxcvz/Report.pdf)

- [printed-circuit-board](https://dgreports.deci.ai/detection/RF100_printed-circuit-board/Report.pdf)

- [radio-signal](https://dgreports.deci.ai/detection/RF100_radio-signal/Report.pdf)

- [road-signs-6ih4y](https://dgreports.deci.ai/detection/RF100_road-signs-6ih4y/Report.pdf)

- [road-traffic](https://dgreports.deci.ai/detection/RF100_road-traffic/Report.pdf)

- [robomasters-285km](https://dgreports.deci.ai/detection/RF100_robomasters-285km/Report.pdf)

- [secondary-chains](https://dgreports.deci.ai/detection/RF100_secondary-chains/Report.pdf)

- [sedimentary-features-9eosf](https://dgreports.deci.ai/detection/RF100_sedimentary-features-9eosf/Report.pdf)

- [shark-teeth-5atku](https://dgreports.deci.ai/detection/RF100_shark-teeth-5atku/Report.pdf)

- [sign-language-sokdr](https://dgreports.deci.ai/detection/RF100_sign-language-sokdr/Report.pdf)

- [signatures-xc8up](https://dgreports.deci.ai/detection/RF100_signatures-xc8up/Report.pdf)

- [smoke-uvylj](https://dgreports.deci.ai/detection/RF100_smoke-uvylj/Report.pdf)

- [soccer-players-5fuqs](https://dgreports.deci.ai/detection/RF100_soccer-players-5fuqs/Report.pdf)

- [soda-bottles](https://dgreports.deci.ai/detection/RF100_soda-bottles/Report.pdf)

- [solar-panels-taxvb](https://dgreports.deci.ai/detection/RF100_solar-panels-taxvb/Report.pdf)

- [stomata-cells](https://dgreports.deci.ai/detection/RF100_stomata-cells/Report.pdf)

- [street-work](https://dgreports.deci.ai/detection/RF100_street-work/Report.pdf)

- [tabular-data-wf9uh](https://dgreports.deci.ai/detection/RF100_tabular-data-wf9uh/Report.pdf)

- [team-fight-tactics](https://dgreports.deci.ai/detection/RF100_team-fight-tactics/Report.pdf)

- [thermal-cheetah-my4dp](https://dgreports.deci.ai/detection/RF100_thermal-cheetah-my4dp/Report.pdf)

- [thermal-dogs-and-people-x6ejw](https://dgreports.deci.ai/detection/RF100_thermal-dogs-and-people-x6ejw/Report.pdf)

- [trail-camera](https://dgreports.deci.ai/detection/RF100_trail-camera/Report.pdf)

- [truck-movement](https://dgreports.deci.ai/detection/RF100_truck-movement/Report.pdf)

- [tweeter-posts](https://dgreports.deci.ai/detection/RF100_tweeter-posts/Report.pdf)

- [tweeter-profile](https://dgreports.deci.ai/detection/RF100_tweeter-profile/Report.pdf)

- [underwater-objects-5v7p8](https://dgreports.deci.ai/detection/RF100_underwater-objects-5v7p8/Report.pdf)

- [underwater-pipes-4ng4t](https://dgreports.deci.ai/detection/RF100_underwater-pipes-4ng4t/Report.pdf)

- [uno-deck](https://dgreports.deci.ai/detection/RF100_uno-deck/Report.pdf)

- [valentines-chocolate](https://dgreports.deci.ai/detection/RF100_valentines-chocolate/Report.pdf)

- [vehicles-q0x2v](https://dgreports.deci.ai/detection/RF100_vehicles-q0x2v/Report.pdf)

- [wall-damage](https://dgreports.deci.ai/detection/RF100_wall-damage/Report.pdf)

- [washroom-rf1fa](https://dgreports.deci.ai/detection/RF100_washroom-rf1fa/Report.pdf)

- [weed-crop-aerial](https://dgreports.deci.ai/detection/RF100_weed-crop-aerial/Report.pdf)

- [wine-labels](https://dgreports.deci.ai/detection/RF100_wine-labels/Report.pdf)

- [x-ray-rheumatology](https://dgreports.deci.ai/detection/RF100_x-ray-rheumatology/Report.pdf)

</details>


<details>

<summary><h3>Segmentation</h3></summary>

- [COCO](https://dgreports.deci.ai/segmentation/COCO/Report.pdf)

- [Cityspace](https://dgreports.deci.ai/segmentation/Cityspace/Report.pdf)

- [VOC](https://dgreports.deci.ai/segmentation/VOC/Report.pdf)

</details>

## Community
<table style="border: 0">
  <tr>
    <td><img src="https://github.com/Deci-AI/data-gradients/raw/master/documentation/assets/discord.png" width="60pt"></td>
    <td><a href="https://discord.gg/2v6cEGMREN"> Click here to join our Discord Community</a></td>
  </tr>
</table>

## License

This project is released under the [Apache 2.0 license](https://dgreports.deci.ai/detection/LICENSE.md).
