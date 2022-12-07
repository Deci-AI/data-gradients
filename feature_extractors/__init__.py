from feature_extractors.segmentation import *
from feature_extractors.common import *

FEATURE_EXTRACTORS = {
    "SegmentationCountNumObjects": SegmentationCountNumObjects,
    "SegmentationGetClassDistribution": SegmentationGetClassDistribution,
    "SegmentationCountSmallObjects": SegmentationCountSmallObjects,
    "NumberOfImagesLabels": NumberOfImagesLabels,
    "NumberOfUniqueClasses": NumberOfUniqueClasses,
    "ImagesResolutions": ImagesResolutions,
    "ImagesAspectRatios": ImagesAspectRatios,
    "AverageBrightness": AverageBrightness,
    "AverageContrast": AverageContrast,
    "NumberOfBackgroundImages": NumberOfBackgroundImages
}
