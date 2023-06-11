"""This is an example will require you to install the super-gradients package.

It shows how DataGradients can be used on top of Datasets provided by SuperGradients.

Required step:
```
pip install super-gradients
```
"""

# Note: This example will require you to install the super-gradients package
from super_gradients.training.dataloaders.dataloaders import coco2017_train, coco2017_val
from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.config.data_config import DetectionDataConfig


if __name__ == "__main__":

    train_loader = coco2017_train()
    val_loader = coco2017_val()

    images_extractor = lambda x: x[0]
    labels_extractor = lambda x: x[1]
    data_config = DetectionDataConfig()

    analyzer = DetectionAnalysisManager(
        data_config=data_config,
        report_title="Testing Data-Gradients2",
        train_data=val_loader,
        val_data=val_loader,
        class_names=val_loader.dataset.classes,
        batches_early_stop=20,
    )

    analyzer.run()
