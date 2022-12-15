from src.managers.segmentation_manager import SegmentationAnalysisManager
from data_loaders.get_torch_loaders import train_data_iterator, val_data_iterator, num_classes, ignore_labels


da = SegmentationAnalysisManager(num_classes=num_classes,
                                 train_data=train_data_iterator,
                                 val_data=val_data_iterator,
                                 ignore_labels=ignore_labels)
da.run()
