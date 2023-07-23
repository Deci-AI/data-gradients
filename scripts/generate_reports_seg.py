from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager


if __name__ == "__main__":

    analyzer = SegmentationAnalysisManager.from_coco(root_dir="/data/coco", year=2017, report_title="COCO")
    analyzer.run()

    analyzer = SegmentationAnalysisManager.from_voc(root_dir="/data/voc/VOCdevkit", year=2012, report_title="VOC")
    analyzer.run()

    from super_gradients.training.dataloaders import cityscapes_train, cityscapes_val

    trainset = cityscapes_train()
    val_set = cityscapes_val()
    # Running on all the Roboflow100 datasets
    analyzer = SegmentationAnalysisManager(
        train_data=trainset,
        val_data=val_set,
        report_title="Cityspace",
        class_names=trainset.dataset.classes + ["Ignore"],
    )
    analyzer.run()
