## Feature Configuration
 
The feature configuration allows you to chose what feature to use, and to adjust their parameters to your needs. 
Follow the steps below to create a YAML configuration file

### 1. YAML Configuration Structure

The configuration file should have the following structure

```yaml
report_sections:
  - name: Section Name
    features:
      - FeatureName1
      - FeatureName2
      - FeatureName3
```

- `report_sections`: A list of sections that will appear in the final report. Each section consists of a name and a list of features to be included.
- `name`: The name of the section.
- `features`: The list of feature names to be included in the section.

### 2. Available Features

Please refer to the default configuration files to explore the available features and their names.
* [Object Detection features](../src/data_gradients/config/detection.yaml) - located in `src/data_gradients/config/detection.yaml` 
* [Semantic Segmentation features](../src/data_gradients/config/segmentation.yaml) - located in `src/data_gradients/config/segmentation.yaml`

For a more in-depth explanation of each feature, please check out [this page](feature_description.md).

### 3. Feature Customization

Each feature can be customized by providing additional arguments in the configuration. Here's an example of customizing the `DetectionSampleVisualization` feature:

```yaml
report_sections:
  - name: Object Detection Features
    features:
      - DetectionSampleVisualization:
          n_rows: 6
          n_cols: 2
          stack_splits_vertically: true
      - ... # Add any other features here
```

- `DetectionSampleVisualization`: The feature name.
- `n_rows`, `n_cols`: The number of rows and columns to use for displaying samples.
- `stack_splits_vertically`: Whether to show train/test samples vertically or side by side.

### 4. Using the Configuration

To use the configuration, provide the path of your YAML file to the relevant analysis manager. 
For example, for object detection analysis:

```python
from data_gradients.managers.detection_manager import DetectionAnalysisManager

train_data = ...
val_data = ...
class_names = ...

analyzer = DetectionAnalysisManager(
    report_title="Testing Data-Gradients Object Detection",
    train_data=train_data,
    val_data=val_data,
    class_names=class_names,
    config_path="path/to/custom-detection-config.yaml"  # Add this parameter to the manager initialization
)

analyzer.run()
```

By following these steps, you can easily customize the features included in your analysis report and fine-tune their parameters according to your requirements.
