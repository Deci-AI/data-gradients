from src import SegmentationAnalysisManager
from data_loaders.get_torch_loaders import train_data_iterator, val_data_iterator, num_classes, ignore_labels


da = SegmentationAnalysisManager(train_data=train_data_iterator,
                                 val_data=val_data_iterator,
                                 num_classes=num_classes,
                                 ignore_labels=ignore_labels)
da.run()
