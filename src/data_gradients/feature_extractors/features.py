from enum import Enum


class ImageFeatures(Enum):
    ImageId = "ImageId"

    ImageWidth = "ImageWidth"
    ImageHeight = "ImageHeight"
    ImageArea = "ImageArea"
    ImageAspectRatio = "ImageAspectRatio"

    ImageMean = "ImageMean"
    ImageStd = "ImageStd"

    ImageAvgBrightness = "ImageAvgBrightness"
    ImageMinBrightness = "ImageMinBrightness"
    ImageMaxBrightness = "ImageMaxBrightness"


class BoundingBoxFeatures(Enum):
    ImageId = "ImageId"

    BoundingBoxId = "BoundingBoxId"
    BoundingBoxLabel = "BoundingBoxLabel"
    BoundingBoxIsCrowd = "BoundingBoxIsCrowd"

    BoundingBoxWidth = "BoundingBoxWidth"
    BoundingBoxHeight = "BoundingBoxHeight"
    BoundingBoxAspectRatio = "BoundingBoxAspectRatio"
    BoundingBoxArea = "BoundingBoxArea"
    BoundingBoxAreaOutsideImage = "BoundingBoxAreaOutsideImage"
    BoundingBoxCenterX = "BoundingBoxCenterX"
    BoundingBoxCenterY = "BoundingBoxCenterY"

    BoundingBoxMaxOverlap = "BoundingBoxMaxOverlap"


class SegmentationMaskFeatures(Enum):
    ImageId = "ImageId"

    SegmentationMaskId = "SegmentationMaskId"
    SegmentationMaskLabel = "SegmentationMaskLabel"

    SegmentationMaskArea = "SegmentationMaskArea"
    SegmentationMaskPerimeter = "SegmentationMaskPerimeter"
    SegmentationMaskCenterOfMassX = "SegmentationMaskCenterOfMassX"
    SegmentationMaskCenterOfMassY = "SegmentationMaskCenterOfMassY"
    SegmentationMaskConvexity = "SegmentationMaskConvexity"
    SegmentationMaskMaxOverlap = "SegmentationMaskMaxOverlap"

    SegmentationMaskWidth = "SegmentationMaskWidth"
    SegmentationMaskHeight = "SegmentationMaskHeight"

    SegmentationMaskDisappearAfterErosion = "SegmentationMaskDisappearAfterErosion"
