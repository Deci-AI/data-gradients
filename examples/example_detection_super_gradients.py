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
from data_gradients.config.interactive_config import DetectionInteractiveConfig


if __name__ == "__main__":

    train_loader = coco2017_train()
    val_loader = coco2017_val()

    images_extractor = lambda x: x[0]
    labels_extractor = lambda x: x[1]
    config = DetectionInteractiveConfig(
        caching_path="coco2017_config.json", reset_cache=False, images_extractor=images_extractor, labels_extractor=labels_extractor
    )
    analyzer = DetectionAnalysisManager(
        config=config,
        report_title="Testing Data-Gradients",
        train_data=train_loader,
        val_data=val_loader,
        class_names=val_loader.dataset.classes,
        batches_early_stop=20,
    )

    analyzer.run()
