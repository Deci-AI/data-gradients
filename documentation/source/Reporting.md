# Report Generation

After obtaining FeaturesCollection object, we can generate a report from it. 
Depending on the dataset type, some of the features might not be available. 
For example, if the dataset is not a segmentation dataset, then segmentation features will not be available. 

Therefore, we need to include in our report only relevant features first and then use one of writers to generate a report.

```python
from data_gradients.managers import SegmentationAnalysisManager
from data_gradients.reports import ReportTemplate 
from data_gradients.writers import NotebookWriter
datasets = ...

results = SegmentationAnalysisManager.extract_features_from_splits(datasets)
report_template = ReportTemplate.get_report_template_with_valid_widgets(results)
NotebookWriter().write_report(results, report_template)
```


## Output writers

| Writer Class      | Description                         |
|-------------------|-------------------------------------|
| MarkdownWriter    | Writes report to a Markdown file    |
| HTMLWriter        | Writes report to a HTML  file       |
| PDFWriter         | Writes report to a PDF file         |
| TensorboardWriter | Writes report to a Tensorboard log  |
| NotebookWriter    | Writes report to a Jupyter Notebook |
