from segmentation import *
from common import *

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
