from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from super_gradients.training.dataloaders import cityscapes_train, cityscapes_val


if __name__ == "__main__":

    analyzer = SegmentationAnalysisManager.from_coco(root_dir="/data/coco", year=2017, report_title="SEG - COCO", batches_early_stop=1000)
    analyzer.run()

    analyzer = SegmentationAnalysisManager.from_voc(root_dir="/data/voc/VOCdevkit", year=2012, report_title="SEG - VOC")
    analyzer.run()

    trainset = cityscapes_train()
    val_set = cityscapes_val()
    analyzer = SegmentationAnalysisManager(
        train_data=trainset,
        val_data=val_set,
        report_title="SEG - Cityspace",
        class_names=trainset.dataset.classes + ["Ignore"],
    )
    analyzer.run()
